import gym
import numpy as np

np.random.seed(0)
env = gym.make('FrozenLake-v0')
env = env.unwrapped

print(env.observation_space)
print(env.action_space)
print(env.unwrapped.P[4][2])


def play_policy(env, policy, render=False):
    total_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


random_policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA

episode_rewards = [play_policy(env, random_policy) for _ in range(100)]
print("随机策略 平均奖励 {}".format(np.mean(episode_rewards)))


## v->q
def v2q(env, v, s=None, gamma=1.):  # 根据状态价值函数计算动作价值函数
    if s is not None:  # 针对单个状态求解
        q = np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q[a] += prob * \
                        (reward + gamma * v[next_state] * (1. - done))
    else:  # 针对所有状态求解
        q = np.zeros((env.unwrapped.nS, env.unwrapped.nA))
        for s in range(env.unwrapped.nS):
            q[s] = v2q(env, v, s, gamma)
    return q


def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    v = np.zeros(env.unwrapped.nS)  # 初始化状态价值函数
    while True:  # 循环
        delta = 0
        for s in range(env.unwrapped.nS):
            vs = sum(policy[s] * v2q(env, v, s, gamma))  # 更新状态价值函数
            delta = max(delta, abs(v[s] - vs))  # 更新最大误差
            v[s] = vs  # 更新状态价值函数
        if delta < tolerant:  # 查看是否满足迭代条件
            break
    return v


print('状态价值函数：')
v_random = evaluate_policy(env, random_policy)
print(v_random.reshape(4, 4))

print('动作价值函数：')
q_random = v2q(env, v_random)
print(q_random)


def improve_policy(env, v, policy, gamma=1.):
    optimal = True
    for s in range(env.unwrapped.nS):
        q = v2q(env, v, s, gamma)
        a = np.argmax(q)
        if policy[s][a] != 1.:
            optimal = False
            policy[s] = 0.
            policy[s][a] = 1.
    return optimal


policy = random_policy.copy()
optimal = improve_policy(env, v_random, policy)
if optimal:
    print('无更新，最优策略为：')
else:
    print('有更新，更新后的策略为：')
print(policy)


def iterate_policy(env, gamma=1., tolerant=1e-6):
    # 初始化为任意一个策略
    policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) \
             / env.unwrapped.nA
    while True:
        v = evaluate_policy(env, policy, gamma, tolerant)  # 策略评估
        if improve_policy(env, v, policy):  # 策略改进
            break
    return policy, v


policy_pi, v_pi = iterate_policy(env)
print('状态价值函数 =')
print(v_pi.reshape(4, 4))
print('最优策略 =')
print(np.argmax(policy_pi, axis=1).reshape(4, 4))

episode_rewards = [play_policy(env, policy_pi) for _ in range(100)]
print("策略迭代 平均奖励：{}".format(np.mean(episode_rewards)))


def iterate_value(env, gamma=1, tolerant=1e-6):
    v = np.zeros(env.unwrapped.nS)  # 初始化
    while True:
        delta = 0
        for s in range(env.unwrapped.nS):
            vmax = max(v2q(env, v, s, gamma))  # 更新价值函数
            delta = max(delta, abs(v[s] - vmax))
            v[s] = vmax
        if delta < tolerant:  # 满足迭代需求
            break

    policy = np.zeros((env.unwrapped.nS, env.unwrapped.nA))  # 计算最优策略
    for s in range(env.unwrapped.nS):
        a = np.argmax(v2q(env, v, s, gamma))
        policy[s][a] = 1.
    return policy, v


policy_vi, v_vi = iterate_value(env)
print('状态价值函数 =')
print(v_vi.reshape(4, 4))
print('最优策略 =')
print(np.argmax(policy_vi, axis=1).reshape(4, 4))

episode_rewards = [play_policy(env, policy_vi) for _ in range(100)]
print("价值迭代 平均奖励：{}".format(np.mean(episode_rewards)))
