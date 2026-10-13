"""Small rank-zero decisions prevent divergent greedy, beam and MTP histories."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from eole.models.model import MTPDraftingMixin
from eole.predict.beam_search import BeamSearchBase, GNMTGlobalScorer
from eole.predict.greedy_search import GreedySearchLM
from eole.utils.tensor_parallel import broadcast_from_rank_zero


class TestTensorParallelDecisions(unittest.TestCase):
    def greedy(self, parallel_gpu):
        strategy = GreedySearchLM(
            pad=0,
            bos=1,
            eos=3,
            unk=2,
            start=1,
            n_best=1,
            batch_size=1,
            global_scorer=GNMTGlobalScorer(0, 0, "none", "none"),
            min_length=0,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            return_attention=False,
            max_length=5,
            temperature=1,
            top_k=1,
            top_p=0,
            beam_size=1,
            ban_unk_token=False,
        )
        strategy.initialize(torch.ones(1, 1, dtype=torch.long), torch.ones(1, dtype=torch.long))
        strategy.parallel_gpu = parallel_gpu
        return strategy

    def test_single_gpu_and_data_parallel_do_not_broadcast(self):
        strategy = self.greedy(1)
        with patch("torch.distributed.broadcast") as broadcast:
            strategy.advance(torch.tensor([[-5.0, -2.0, -3.0, -1.0]]), None)
        broadcast.assert_not_called()
        self.assertEqual(strategy.alive_seq[0, -1].item(), 3)

    def test_greedy_uses_root_token_score_and_eos_decision(self):
        strategy = self.greedy(2)

        def root_choice(tensor, src):
            self.assertEqual(src, 0)
            tensor.fill_(3 if tensor.dtype == torch.long else -0.75)

        # The local best token is 1, but the root selected EOS=3.
        with patch("torch.distributed.broadcast", side_effect=root_choice) as broadcast:
            strategy.advance(torch.tensor([[-5.0, -1.0, -3.0, -2.0]]), None)
        self.assertEqual(broadcast.call_count, 2)
        self.assertEqual(strategy.alive_seq[0, -1].item(), 3)
        self.assertEqual(strategy.is_finished_list, [[True]])
        self.assertEqual(strategy.beams_scores.item(), -0.75)

    def test_beam_parent_indices_and_scores_are_shared(self):
        strategy = SimpleNamespace(beam_size=2, parallel_gpu=2, target_prefixing=lambda x: x)

        def root_choice(tensor, src):
            chosen = torch.tensor([[7, 0]]) if tensor.dtype == torch.long else torch.tensor([[-0.2, -0.7]])
            tensor.copy_(chosen)

        with patch("torch.distributed.broadcast", side_effect=root_choice):
            scores, ids = BeamSearchBase._pick(strategy, torch.arange(8).float().reshape(2, 4))
        torch.testing.assert_close(ids, torch.tensor([[7, 0]]))
        torch.testing.assert_close(scores, torch.tensor([[-0.2, -0.7]]))
        # Flat index 7 means parent beam 1, token 3; both must agree.
        self.assertEqual((ids // 4).tolist(), [[1, 0]])

    def test_speculative_scores_are_shared_for_bulk_advance(self):
        strategy = self.greedy(2)
        with patch("torch.distributed.broadcast", side_effect=lambda tensor, src: tensor.fill_(-0.5)):
            ids = strategy.advance_speculative(torch.tensor([[1, 2]]), torch.zeros(1, 2, 4), 2, False)
        torch.testing.assert_close(ids, torch.tensor([[1, 2]]))
        self.assertEqual(strategy.beams_scores.item(), -1)

    def test_mtp_first_draft_uses_root_token(self):
        model = MTPDraftingMixin()
        head = nn.Identity()
        head.emb_norm = nn.Identity()
        head.layer = SimpleNamespace(self_attn=SimpleNamespace(parallel_gpu=2))

        class Head(nn.Module):
            emb_norm = head.emb_norm
            layer = head.layer

            def forward(self, hidden, embedding, **kwargs):
                return hidden

        model.mtp_heads = [Head()]
        model.decoder = SimpleNamespace(rope=None)
        model.tgt_emb = SimpleNamespace(embeddings=nn.Embedding(4, 4))
        model.generator = lambda x: torch.tensor([[9.0, 0.0, 0.0, 0.0]])
        with patch("torch.distributed.broadcast", side_effect=lambda tensor, src: tensor.fill_(2)):
            drafts = model.draft_mtp_tokens(torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.long), 0)
        self.assertEqual(drafts[0].item(), 2)

    def test_helper_preserves_tensor_storage(self):
        tensor = torch.zeros(2, dtype=torch.long)
        with patch("torch.distributed.broadcast") as broadcast:
            result = broadcast_from_rank_zero(tensor, 2)
        self.assertIs(result, tensor)
        broadcast.assert_called_once_with(tensor, src=0)
