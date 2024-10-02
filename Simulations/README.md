singleEnv: ultimate goal is to achieve RL training for pack of 3 predator (controlled by one neural network) to catch randomly moving prey using PPO policy
    1. Single predator and stationary prey + second trial with randomly moving prey
        -   Train for 1_500_000 total timesteps, with 1500 timesteps max per episode
        -   Hyperparameters: modifed learning rate, lowering it (0.0002 vs default of 0.0005) to 
            encourage  more exploitation of known strategies rather than excessive exploration.
        -   Reward and punishment systems:
            -   

            !!! Note that it is important to have a symmetrical action space to elliminate bias towards  
                certain actions. Ie [-1, 1] instead of [0, 2]. In fact, Stable Baselines3 only accepts [-1, 1] as output range for continuous action spaces, else the actions are automaticaly clipped or normalized.
                
multiEnv: RL training pack of 3 predators to catch a prey, simultaneously being trained to evade predators, also using A2C policy

1. Made into discrete action spaces
2. Decreased the amount of degrees rotated per step to 1 to allow more fine tuning of position
3. Add reward for small deviation from desired course and reward for going straight while closing in (discourage circling behaviour)
4. Increase threshold for what counts as 'closing in' have to be 0.5 units or more closer than last step, increased from 0.25. In addition, provide more reward for going straight and increase range for angle deviation (to make sure reward for going straight is more frequent)