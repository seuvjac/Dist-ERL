"""Policy template and weight loading tests."""

from src.utils.policy_utils import build_model_template
from src.utils.policies import DDPGPolicy, TD3Policy


def test_template_matches_ddpg_actor():
    sd, ad = 17, 6
    template = build_model_template(sd, ad, algorithm='DDPG')
    policy = DDPGPolicy(state_dim=sd, action_dim=ad)
    for key in policy.actor.state_dict().keys():
        full = f'actor.{key}'
        assert full in template
    for key in policy.critic.state_dict().keys():
        full = f'critic.{key}'
        assert full in template


def test_template_matches_td3_actor():
    sd, ad = 17, 6
    template = build_model_template(sd, ad, algorithm='TD3')
    policy = TD3Policy(state_dim=sd, action_dim=ad)
    for key in policy.actor.state_dict().keys():
        assert f'actor.{key}' in template
    for key in policy.critic1.state_dict().keys():
        assert f'critic1.{key}' in template
