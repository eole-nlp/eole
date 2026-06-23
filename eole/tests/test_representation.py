import unittest

import torch

from eole.modules.representation import RepresentationExtractor


class TestRepresentationExtractor(unittest.TestCase):
    def test_token_embeddings_returns_last_layer(self):
        hidden_states = [torch.full((1, 2, 3), 1.0), torch.full((1, 2, 3), 2.0)]
        extractor = RepresentationExtractor(layer="last")

        result = extractor.token_embeddings(hidden_states, torch.ones(1, 2))

        self.assertTrue(torch.equal(result, hidden_states[-1]))

    def test_token_embeddings_returns_configured_layer(self):
        hidden_states = [torch.full((1, 2, 3), 1.0), torch.full((1, 2, 3), 2.0)]
        extractor = RepresentationExtractor(layer="0")

        result = extractor.token_embeddings(hidden_states, torch.ones(1, 2))

        self.assertTrue(torch.equal(result, hidden_states[0]))

    def test_forward_cls_pool_uses_token_embeddings_first_token(self):
        hidden_states = [
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        ]
        extractor = RepresentationExtractor(layer="last", pool="cls")

        result = extractor(hidden_states, torch.ones(1, 2))

        self.assertTrue(torch.equal(result, torch.tensor([[5.0, 6.0]])))


if __name__ == "__main__":
    unittest.main()
