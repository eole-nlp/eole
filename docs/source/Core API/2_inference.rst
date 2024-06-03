Prediction
==================

Predictions
-------------

.. autoclass:: eole.predict.prediction.Prediction
    :members:

.. autoclass:: eole.predict.prediction.PredictionBuilder
    :members:

Predictor Classes
-----------------

.. autoclass:: eole.predict.inference.Inference
    :members:

.. autoclass:: eole.predict.Translator
    :members:

.. autoclass:: eole.predict.GeneratorLM
    :members:

.. autoclass:: eole.predict.Encoder
    :members:


Decoding Strategies
--------------------
.. autoclass:: eole.predict.DecodeStrategy
    :members:

.. autoclass:: eole.predict.BeamSearch
    :members:

.. autofunction:: eole.predict.greedy_search.sample_with_temperature

.. autoclass:: eole.predict.GreedySearch
    :members:

Scoring
--------
.. autoclass:: eole.predict.penalties.PenaltyBuilder
    :members:

.. autoclass:: eole.predict.GNMTGlobalScorer
    :members:
