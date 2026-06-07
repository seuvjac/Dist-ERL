"""Policy template and weight loading tests."""

from src.utils.policy_utils import build_model_template
from src.utils.policies import DDPGPolicy, FSACPolicy, PPOPolicy, TD3Policy


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


def test_template_matches_fsac_actor_only():
    sd, ad = 4, 2
    template = build_model_template(sd, ad, algorithm='FSAC')
    policy = FSACPolicy(state_dim=sd, action_dim=ad)
    for key, value in policy.actor.state_dict().items():
        full = f'actor.{key}'
        assert full in template
        assert tuple(template[full].shape) == tuple(value.shape)
    assert not any(key.startswith('critic') for key in template)


def test_template_matches_discrete_ppo_actor():
    sd, ad = 6, 3
    template = build_model_template(sd, ad, algorithm='PPO', discrete=True)
    policy = PPOPolicy(state_dim=sd, action_dim=ad, discrete=True)
    for key, value in policy.actor.state_dict().items():
        full = f'actor.{key}'
        assert full in template
        assert tuple(template[full].shape) == tuple(value.shape)
    for key, value in policy.critic.state_dict().items():
        full = f'critic.{key}'
        assert full in template
        assert tuple(template[full].shape) == tuple(value.shape)


def test_template_matches_continuous_ppo_actor():
    sd, ad = 8, 2
    template = build_model_template(sd, ad, algorithm='PPO', discrete=False)
    policy = PPOPolicy(state_dim=sd, action_dim=ad, discrete=False)
    for key, value in policy.actor.state_dict().items():
        full = f'actor.{key}'
        assert full in template
        assert tuple(template[full].shape) == tuple(value.shape)
    for key, value in policy.critic.state_dict().items():
        full = f'critic.{key}'
        assert full in template
        assert tuple(template[full].shape) == tuple(value.shape)
