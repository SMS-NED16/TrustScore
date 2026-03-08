"""
TrustScore Recommender -- Judge and Model recommendation engine.

Two flows:
  - recommend_judges(): multi-source rank aggregation for judge selection
  - recommend_model(): direct benchmark lookup via hierarchical taxonomy
"""

from recommender.service import recommend_judges, recommend_model

__all__ = ["recommend_judges", "recommend_model"]
