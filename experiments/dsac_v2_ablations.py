import sys
from pathlib import Path

DSAC_ROOT = Path(__file__).resolve().parent.parent / "DSAC-v2"
sys.path.insert(0, str(DSAC_ROOT))
sys.path.insert(0, str(DSAC_ROOT / "utils"))

import torch
from torch.nn.functional import huber_loss
from dsac_v2 import DSAC_V2

class DSAC_V2_NoExpectedValue(DSAC_V2):
    def _DSAC_V2__compute_target_q(self, r, done, q, q_std, q_next, q_next_sample, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
            q_next_sample - self._DSAC_V2__get_alpha() * log_prob_a_next
        )
        target_q_sample = r + (1 - done) * self.gamma * (
            q_next_sample - self._DSAC_V2__get_alpha() * log_prob_a_next
        )
        td_bound = 3 * q_std
        difference = torch.clamp(target_q_sample - q, -td_bound, td_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

class DSAC_V2_SingleQ(DSAC_V2):
    def _DSAC_V2__compute_loss_q(self, data):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

        q1, q1_std, _ = self._DSAC_V2__q_evaluate(obs, act, self.networks.q1)

        if self.mean_std1 == -1.0:
            self.mean_std1 = torch.mean(q1_std.detach())
        else:
            self.mean_std1 = (1 - self.tau_b) * self.mean_std1 + self.tau_b * torch.mean(q1_std.detach())


        q1_next, _, q1_next_sample = self._DSAC_V2__q_evaluate(
            obs2, act2, self.networks.q1_target
        )
        # no min use q1 instead of min(q1_next, q2_next)
        q_next = q1_next
        q_next_sample = q1_next_sample

        target_q1, target_q1_bound = self._DSAC_V2__compute_target_q(
            rew,
            done,
            q1.detach(),
            self.mean_std1.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        q1_std_detach = torch.clamp(q1_std, min=0.).detach()
        bias = 0.1

        ratio1 = (torch.pow(self.mean_std1, 2) / (torch.pow(q1_std_detach, 2) + bias)).clamp(min=0.1, max=10)

        q1_loss = torch.mean(ratio1 *(huber_loss(q1, target_q1, delta = 50, reduction='none')
                                      + q1_std *(q1_std_detach.pow(2) - huber_loss(q1.detach(), target_q1_bound, delta = 50, reduction='none'))/(q1_std_detach +bias)
                            ))
        zero = torch.tensor(0.0)
        return q1_loss, q1.detach().mean(), zero, q1_std.detach().mean(), zero, q1_std.min().detach(), zero

    def _DSAC_V2__compute_loss_policy(self, data):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        q1, _, _ = self._DSAC_V2__q_evaluate(obs, new_act, self.networks.q1)
        loss_policy = (self._DSAC_V2__get_alpha() * new_log_prob - q1).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy