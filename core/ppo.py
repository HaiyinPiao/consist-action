import torch


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, repeats, fixed_log_probs, fixed_rpt_log_probs, clip_epsilon, l2_reg):

    """update critic"""
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update policy"""
    action_log_probs, repeat_log_probs = policy_net.get_log_prob(states, actions, repeats)
    action_ratio = torch.exp(action_log_probs - fixed_log_probs)
    repeat_ratio = torch.exp(repeat_log_probs - fixed_rpt_log_probs)
    ratio = action_ratio + 1e-1*repeat_ratio
    # ratio = action_ratio
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()
    # ratio = torch.exp(log_probs - fixed_log_probs)
    # surr1 = ratio * advantages
    # surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    # policy_surr = -torch.min(surr1, surr2).mean()
    # optimizer_policy.zero_grad()
    # policy_surr.backward()
    # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    # optimizer_policy.step()
