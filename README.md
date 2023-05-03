Download Link: https://assignmentchef.com/product/solved-cs294-112-deep-reinforcement-learning-hw2-policy-gradients
<br>
<h1>1      Introduction</h1>

The goal of this assignment is to experiment with policy gradient and its variants, including variance reduction methods. Your goals will be to set up policy gradient for both continuous and discrete environments and experiment with variance reduction tricks, including implementing reward-to-go and neural network baselines.

Turn in your report and code for the full homework as described in Problem 2 by September 19, 2018.

<h1>2      Review</h1>

Recall that the reinforcement learning objective is to learn a <em>θ</em><sup>∗ </sup>that maximizes the objective function:

<em>J</em>(<em>θ</em>) = E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[<em>r</em>(<em>τ</em>)]                                                         (1)

where

<em>T</em>

<em>π<sub>θ</sub></em>(<em>τ</em>) = <em>p</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub><em>,…,s<sub>T</sub>,a<sub>T</sub></em>) = <em>p</em>(<em>s</em><sub>1</sub>)<em>π<sub>θ</sub></em>(<em>a</em><sub>1</sub>|<em>s</em><sub>1</sub>)<sup>Y</sup><em>p</em>(<em>s<sub>t</sub></em>|<em>s<sub>t</sub></em><sub>−1</sub><em>,a<sub>t</sub></em><sub>−1</sub>)<em>π<sub>θ</sub></em>(<em>a<sub>t</sub></em>|<em>s<sub>t</sub></em>)

<em>t</em>=2

and

<em>T</em>

<em>r</em>(<em>τ</em>) = <em>r</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub><em>,…,s<sub>T</sub>,a<sub>T</sub></em>) = <sup>X</sup><em>r</em>(<em>s<sub>t</sub>,a<sub>t</sub></em>)<em>.</em>

<em>t</em>=1

The policy gradient approach is to directly take the gradient of this objective:

Z

∇<em><sub>θ</sub>J</em>(<em>θ</em>) = ∇<em><sub>θ                </sub>π<sub>θ</sub></em>(<em>τ</em>)<em>r</em>(<em>τ</em>)<em>dτ                                                         </em>(2)

Z

=        <em>π<sub>θ</sub></em>(<em>τ</em>)∇<em><sub>θ </sub></em>log<em>π<sub>θ</sub></em>(<em>τ</em>)<em>r</em>(<em>τ</em>)<em>dτ.                                           </em>(3)

In practice, the expectation over trajectories <em>τ </em>can be approximated from a batch of <em>N </em>sampled trajectories:

)                                                                 (4)

<em> .                        </em>(5)

Here we see that the policy <em>π<sub>θ </sub></em>is a probability distribution over the action space, conditioned on the state. In the agent-environment loop, the agent samples an action <em>a<sub>t </sub></em>from <em>π<sub>θ</sub></em>(·|<em>s<sub>t</sub></em>) and the environment responds with a reward <em>r</em>(<em>s<sub>t</sub>,a<sub>t</sub></em>).

One way to reduce the variance of the policy gradient is to exploit causality: the notion that the policy cannot affect rewards in the past, yielding following the modified objective, where the sum of rewards here is a sample estimate of the <em>Q </em>function, known as the “reward-togo:”

<em> .                          </em>(6)

Multiplying a discount factor <em>γ </em>to the rewards can be interpreted as encouraging the agent to focus on rewards closer in the future, which can also be thought of as a means for reducing variance (because there is more variance possible futures further into the future). We saw in lecture that the discount factor can be incorporated in two ways.

The first way applies the discount on the rewards from full trajectory:

!

(7)

and the second way applies the discount on the “reward-to-go:”

<em> .                       </em>(8)

.

We have seen in lecture that subtracting a baseline that is a constant with respect to <em>τ </em>from the sum of rewards

∇<em><sub>θ</sub>J</em>(<em>θ</em>) = ∇<em><sub>θ</sub></em>E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[<em>r</em>(<em>τ</em>) − <em>b</em>]                                                   (9)

leaves the policy gradient unbiased because

∇<em><sub>θ</sub></em>E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[<em>b</em>] = E<em>τ</em><sub>∼<em>π</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[∇<em><sub>θ </sub></em>log<em>π<sub>θ</sub></em>(<em>τ</em>) · <em>b</em>] = 0<em>.</em>

In this assignment, we will implement a value function <em>V<sub>φ</sub><sup>π </sup></em>which acts as a <em>state-dependent </em>baseline. The value function is trained to approximate the sum of future rewards starting from a particular state:

<em>T</em>

<em>V<sub>φ</sub><sup>π</sup></em>(<em>s<sub>t</sub></em>) ≈ <sup>X</sup>E<em>π<sub>θ </sub></em>[<em>r</em>(<em>s<sub>t</sub></em>0<em>,a<sub>t</sub></em>0)|<em>s<sub>t</sub></em>]<em>,                                                 </em>(10)

<em>t</em><sup>0</sup>=<em>t</em>

so the approximate policy gradient now looks like this:

<em> .         </em>(11)

<strong>Problem 1. State-dependent baseline: </strong>In lecture we saw that the policy gradient is unbiased if the baseline is a constant with respect to <em>τ </em>(Equation 9). The purpose of this problem is to help convince ourselves that subtracting a state-dependent baseline from the return keeps the policy gradient unbiased. For clarity we will use <em>p<sub>θ</sub></em>(<em>τ</em>) instead of <em>π<sub>θ</sub></em>(<em>τ</em>), although they mean the same thing. Using the <a href="https://en.wikipedia.org/wiki/Law_of_total_expectation">law of iterated expectations</a> we will show that the policy gradient is still unbiased if the baseline <em>b </em>is function of a state at a particular timestep of <em>τ </em>(Equation 11). Recall from equation 3 that the policy gradient can be expressed as

E<em>τ</em><sub>∼<em>p</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[∇<em><sub>θ </sub></em>log<em>p<sub>θ</sub></em>(<em>τ</em>)<em>r</em>(<em>τ</em>)]<em>.</em>

By breaking up <em>p<sub>θ</sub></em>(<em>τ</em>) into dynamics and policy terms, we can discard the dynamics terms, which are not functions of <em>θ</em>:

<em> .</em>

When we subtract a state dependent baseline <em>b</em>(<em>s<sub>t</sub></em>) (recall equation 11) we get

<em> .</em>

Our goal for this problem is to show that

<em>.</em>

By <a href="https://brilliant.org/wiki/linearity-of-expectation/">linearity of expectation</a> we can consider each term in this sum independently, so we can equivalently show that

<em>T</em>

X

E<em>τ</em><sub>∼<em>p</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>[∇<em><sub>θ </sub></em>log<em>π<sub>θ</sub></em>(<em>a<sub>t</sub></em>|<em>s<sub>t</sub></em>)(<em>b</em>(<em>s<sub>t</sub></em>))] = 0<em>.                                        </em>(12)

<em>t</em>=1

<ul>

 <li>Using the chain rule, we can express <em>p<sub>θ</sub></em>(<em>τ</em>) as a product of the state-action marginal (<em>s<sub>t</sub>,a<sub>t</sub></em>) and the probability of the rest of the trajectory conditioned on (<em>s<sub>t</sub>,a<sub>t</sub></em>) (which we denote as (<em>τ/s<sub>t</sub>,a<sub>t</sub></em>|<em>s<sub>t</sub>,a<sub>t</sub></em>)):</li>

</ul>

<em>p<sub>θ</sub></em>(<em>τ</em>) = <em>p<sub>θ</sub></em>(<em>s<sub>t</sub>,a<sub>t</sub></em>)<em>p<sub>θ</sub></em>(<em>τ/s<sub>t</sub>,a<sub>t</sub></em>|<em>s<sub>t</sub>,a<sub>t</sub></em>)

Please show equation 12 by using the law of iterated expectations, breaking E<em>τ</em><sub>∼<em>p</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>by decoupling the state-action marginal from the rest of the trajectory.

<ul>

 <li>Alternatively, we can consider the structure of the MDP and express <em>p<sub>θ</sub></em>(<em>τ</em>) as a product of the trajectory distribution up to <em>s<sub>t </sub></em>(which we denote as (<em>s</em><sub>1:<em>t</em></sub><em>,a</em><sub>1:<em>t</em>−1</sub>)) and the trajectory distribution after <em>s<sub>t </sub></em>conditioned on the first part (which we denote as</li>

</ul>

(<em>s</em><em>t</em>+1:<em>T</em><em>,a</em><em>t</em>:<em>T</em>|<em>s</em>1:<em>t</em><em>,a</em>1:<em>t</em>−1)):

<em>p</em><em>θ</em>(<em>τ</em>) = <em>p</em><em>θ</em>(<em>s</em>1:<em>t</em><em>,a</em>1:<em>t</em>−1)<em>p</em><em>θ</em>(<em>s</em><em>t</em>+1:<em>T</em><em>,a</em><em>t</em>:<em>T</em>|<em>s</em>1:<em>t</em><em>,a</em>1:<em>t</em>−1)

<ul>

 <li>Explain why, for the inner expectation, conditioning on (<em>s</em><sub>1</sub><em>,a</em><sub>1</sub><em>,…,a<sub>t</sub></em>∗<sub>−1</sub><em>,s<sub>t</sub></em>∗) is equivalent to conditioning only on <em>s<sub>t</sub></em>∗.</li>

 <li>Please show equation 12 by using the law of iterated expectations, breaking</li>

</ul>

E<em>τ</em><sub>∼<em>p</em></sub><em><sub>θ</sub></em>(<em><sub>τ</sub></em><sub>) </sub>by decoupling trajectory up to <em>s<sub>t </sub></em>from the trajectory after <em>s<sub>t</sub></em>.

Since the policy gradient with respect to <em>θ </em>can be decoupled as a summation of terms over timesteps <em>t </em>∈ [1<em>,T</em>], because we have shown that the policy gradient is unbiased for each of these terms, the entire policy gradient is also unbiased with respect to a vector of statedependent baselines over the timesteps: [<em>b</em>(<em>s</em><sub>1</sub>)<em>,b</em>(<em>s</em><sub>2</sub>)<em>,…b</em>(<em>s<sub>T</sub></em>)].

<h1>3       Code Setup</h1>

<h2>3.1      Files</h2>

The starter code is available <a href="https://github.com/berkeleydeeprlcourse/homework/tree/master/hw2">here</a><a href="https://github.com/berkeleydeeprlcourse/homework/tree/master/hw2">.</a> The only file you need to modify in this homework is train_pg_f18.py. The files logz.py and plots.py are utility files; while you should look at them to understand their functionality, you will not modify them. For the Lunar Lander task, use the provided lunar_lander.py file instead of gym/envs/box2d/lunar_lander.py. After you fill in the appropriate methods, you should be able to just run python train_pg_f18.py with some command line options to perform the experiments. To visualize the results, you can run python plot.py path/to/logdir.

<h2>3.2       Overview</h2>

The function train_PG is used to perform the actual training for policy gradient. The parameters passed into this function specify the algorithm’s hyperparameters and environment. The Agent class contains methods that define the computation graph, sample trajectories, estimate returns, and update the parameters of the policy.

At a high level, the dataflow of the code is structured like this:

<ol>

 <li><em>Build a static computation graph </em>in Tensorflow.</li>

 <li><em>Set up a Tensorflow session </em>to initialize the parameters of the computation graph. This is the only time you need to set up the session.</li>

</ol>

Then we will repeat Steps 3 through 5 for <em>N </em>iterations:

<ol start="3">

 <li><em>Sample trajectories </em>by executing the Tensorflow op that samples an action given an observation from the environment. Collect the states, actions, and rewards as numpy variables.</li>

 <li><em>Estimate returns </em>in numpy (estimated Q values, baseline predictions, advantages).</li>

 <li><em>Update parameters </em>by executing the Tensorflow op that updates the parameters given what you computed in Step 4.</li>

</ol>

<h1>4         Constructing the computation graph</h1>

<strong>Problem 2. Neural networks: </strong>We will now begin to implement a neural network that parametrizes <em>π<sub>θ</sub></em>.

<ul>

 <li>Implement the utility function, build_mlp, which will build a feedforward neural network with fully connected units (Hint: use layers.dense). Test it to make sure that it produces outputs of the expected size and shape. <strong>You do not need to include anything in your write-up about this, </strong>it will just make your life easier.</li>

 <li>Next, implement the method build_computation_graph. At this point, you only need to implement the parts with the “Problem 2” header.

  <ul>

   <li>Define the placeholder for the advantages in define_placeholders. We have already defined placeholders for the observations and actions. The advantages will correspond to <em>r</em>(<em>τ</em>) in the policy gradient, which may or may not include a subtracted baseline value.</li>

   <li>Create the symbolic operation policyforwardpass: This outputs the parameters of a distribution <em>π<sub>θ</sub></em>(<em>a</em>|<em>s</em>). In this homework, when the distribution is over discrete actions these parameters will be the logits of a categorical distribution, and when the distribution is over continuous actions these parameters will be the mean and the log standard deviation of a multivariate Gaussian distribution. This operation will be an input to Agent.sampleaction and Agent.getlogprob.</li>

   <li>Create the symbolic operation sampleaction: This produces a Tensorflow op, self.sysampledac that samples an action from <em>π<sub>θ</sub></em>(<em>a</em>|<em>s</em>). This operation will be called in Agent.sampletrajectories.</li>

   <li>Create the symbolic operation getlogprob: Given an action that the agent took in the environment, this computes the log probability of that action under <em>π<sub>θ</sub></em>(<em>a</em>|<em>s</em>). This will be used in the loss function.</li>

   <li>In build_computation_graph implement a loss function (which call use the result from Agent.getlogprob) to whose gradient is</li>

  </ul></li>

</ul>

<em>.</em>

<h1>5        Implement Policy Gradient</h1>

<h2>5.1         Implementing the policy gradient loop</h2>

<strong>Problem 3. Policy Gradient: </strong>Recall from lecture that an RL algorithm can viewed as consisting of three parts, which are reflected in the training loop of train_PG:

<ol>

 <li>sample_trajectories: Generate samples (e.g. run the policy to collect trajectories consisting of state transitions (<em>s,a,s</em><sup>0</sup><em>,r</em>))</li>

 <li>estimate_return: Estimate the return (e.g. sum together discounted rewards from the trajectories, or learn a model that predicts expected total future discounted reward)</li>

 <li>update_parameters: Improve the policy (e.g. update the parameters of the policy with policy gradient)</li>

</ol>

In our implementation, for clarity we will update the parameters of the value function baseline also in the third step (Agent.update_parameters), rather than in the second step (as was described in lecture). You only need to implement the parts with the “Problem 3” header.

<ul>

 <li><strong>Sample trajectories: </strong>In sampletrajectories, use the Tensorflow session to call self.sysampledac to sample an action given an observation from the environment.</li>

 <li><strong>Estimate return: </strong>We will now implement <em>r</em>(<em>τ</em>) from Equation 1. Please implement the method sum_of_rewards, which will return a sample estimate of the discounted return, for both the full-trajectory (Equation 7) case, where</li>

</ul>

<em>T r</em>(<em>τ</em><em>i</em>) = X<em>γ</em><em>t</em>0−1<em>r</em>(<em>s</em><em>it,a</em><em>it</em>)

<em>t</em>=1

and for the “reward-to-go” case (Equation 8) where

<em>T r</em>(<em>τ</em><em>i</em>) = X<em>γ</em><em>t</em>0−<em>t</em><em>r</em>(<em>s</em><em>it</em>0<em>,a</em><em>it</em>0)<em>.</em>

<em>t</em><sup>0</sup>=<em>t</em>

In Agent.estimate_return, normalize the advantages to have a mean of zero and a standard deviation of one. This is a trick for reducing variance.

<ul>

 <li><strong>Update parameters: </strong>In update_parameters use the Tensorflow session to call the update operation self.updateop to update the parameters of the policy. You will need to figure out the inputs to feeddict.</li>

</ul>

<h2>5.2       Experiments</h2>

After you have implemented the code, we will run experiments to get a feel for how different settings impact the performance of policy gradient methods.

<strong>Problem 4. CartPole: </strong>Run the PG algorithm in the discrete CartPole-v0 environment from the command line as follows:

python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna –exp_name sb_no_rtg_dna

python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna –exp_name sb_rtg_dna

python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg –exp_name sb_rtg_na

python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna –exp_name lb_no_rtg_dna

python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna –exp_name lb_rtg_dna

python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg –exp_name lb_rtg_na

What’s happening there:

<ul>

 <li>-n : Number of iterations.</li>

 <li>-b : Batch size (number of state-action pairs sampled while acting according to the current policy at each iteration).</li>

 <li>-e : Number of experiments to run with the same configuration. Each experiment will start with a different randomly initialized policy, and have a different stream of random numbers.</li>

 <li>-dna : Flag: if present, sets normalize_advantages to False. Otherwise, by default, normalize_advantages=True.</li>

 <li>-rtg : Flag: if present, sets reward_to_go=True. Otherwise, reward_to_go=False by default.</li>

 <li>–exp_name : Name for experiment, which goes into the name for the data directory.</li>

</ul>

Various other command line arguments will allow you to set batch size, learning rate, network architecture (number of hidden layers and the size of the hidden layers—for CartPole, you can use one hidden layer with 32 units), and more.

<strong>Deliverables for report:</strong>

<ul>

 <li>Graph the results of your experiments <strong>using the plot.py file we provide. </strong>Create two graphs.

  <ul>

   <li>In the first graph, compare the learning curves (average return at each iteration) for the experiments prefixed with sb_. (The small batch experiments.)</li>

   <li>In the second graph, compare the learning curves for the experiments prefixed with lb_. (The large batch experiments.)</li>

  </ul></li>

 <li>Answer the following questions briefly:

  <ul>

   <li>Which gradient estimator has better performance without advantage-centering— the trajectory-centric one, or the one using reward-to-go?</li>

   <li>Did advantage centering help?</li>

   <li>Did the batch size make an impact?</li>

  </ul></li>

 <li>Provide the exact command line configurations you used to run your experiments. (To verify batch size, learning rate, architecture, and so on.)</li>

</ul>

<strong>What to Expect:</strong>

<ul>

 <li>The best configuration of CartPole in both the large and small batch cases converge to a maximum score of 200.</li>

</ul>

<strong>Problem 5. InvertedPendulum: </strong>Run experiments in InvertedPendulum-v2 continuous control environment as follows:

<table width="624">

 <tbody>

  <tr>

   <td width="624">python train_pg_f18.py InvertedPendulum-v2 -ep 1000 –discount 0.9 -n 100 -e 3-l 2 -s 64 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg –exp_name ip_b&lt;b*&gt;_r&lt;r*&gt;</td>

  </tr>

 </tbody>

</table>

where your task is to find the smallest batch size b* and largest learning rate r* that gets to optimum (maximum score of 1000) in less than 100 iterations. The policy performance may fluctuate around 1000 – this is fine. The precision of b* and r* need only be one significant digit.

<strong>Deliverables:</strong>

<ul>

 <li>Given the b* and r* you found, provide a learning curve where the policy gets to optimum (maximum score of 1000) in less than 100 iterations. (This may be for a single random seed, or averaged over multiple.)</li>

 <li>Provide the exact command line configurations you used to run your experiments.</li>

</ul>

<h1>6         Implement Neural Network Baselines</h1>

For the rest of the assignment we will use “reward-to-go.”

<strong>Problem 6. Neural network baseline: </strong>We will now implement a value function as a state-dependent neural network baseline. The sections in the code are marked by “Problem 6.”

<ul>

 <li>In build_computation_graph implement <em>V<sub>φ</sub><sup>π</sup></em>, a neural network that predicts the expected return conditioned on a state. Also implement the loss function to train this network and its update operation self.baselineop.</li>

 <li>In compute_advantage, use the neural network to predict the expected state-conditioned return (use the session to call self.baselineprediction), normalize it to match the statistics of the current batch of “reward-to-go”, and subtract this value from the “reward-to-go” to yield an estimate of the advantage. This implements</li>

 <li>In update_parameters, update the parameters of the the neural network baseline by using the Tensorflow session to call self.baselineop. “Rescale” the target values for the neural network baseline to have a mean of zero and a standard deviation of one.</li>

</ul>

<h1>7       More Complex Tasks</h1>

<strong>Note: </strong>The following tasks would take quite a bit of time to train. Please start early!

<strong>Problem 7: LunarLander </strong>For this problem, you will use your policy gradient implementation to solve LunarLanderContinuous-v2. Use an episode length of 1000. The purpose of this problem is to help you debug your baseline implementation. Run the following command:

python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 –discount 0.99 -n

100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg –nn_baseline –exp_name ll_b40000_r0.005

<strong>Deliverables:</strong>

<ul>

 <li>Plot a learning curve for the above command. You should expect to achieve an average return of around 180.</li>

</ul>

<strong>Problem 8: HalfCheetah </strong>For this problem, you will use your policy gradient implementation to solve HalfCheetah-v2. Use an episode length of 150, which is shorter than the default of 1000 for HalfCheetah (which would speed up your training significantly). Search over batch sizes b ∈ [10000<em>,</em>30000<em>,</em>50000] and learning rates r ∈ [0<em>.</em>005<em>,</em>0<em>.</em>01<em>,</em>0<em>.</em>02] to replace &lt;b&gt; and &lt;r&gt; below:

<table width="624">

 <tbody>

  <tr>

   <td width="624">python train_pg_f18.py HalfCheetah-v2 -ep 150 –discount 0.9 -n 100 -e 3 -l 2-s 32 -b &lt;b&gt; -lr &lt;r&gt; -rtg –nn_baseline –exp_name hc_b&lt;b&gt;_r&lt;r&gt;</td>

  </tr>

 </tbody>

</table>

<strong>Deliverables:</strong>

<ul>

 <li>How did the batch size and learning rate affect the performance?</li>

 <li>Once you’ve found suitable values of b and r among those choices (let’s call them b* and r*), use b* and r* and run the following commands (remember to replace the terms in the angle brackets):</li>

</ul>

<table width="585">

 <tbody>

  <tr>

   <td width="585">python train_pg_f18.py HalfCheetah-v2 -ep 150 –discount 0.95 -n 100 -e 3-l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; –exp_name hc_b&lt;b*&gt;_r&lt;r*&gt; python train_pg_f18.py HalfCheetah-v2 -ep 150 –discount 0.95 -n 100 -e 3-l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg –exp_name hc_b&lt;b*&gt;_r&lt;r*&gt; python train_pg_f18.py HalfCheetah-v2 -ep 150 –discount 0.95 -n 100 -e 3-l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; –nn_baseline –exp_name hc_b&lt;b*&gt;_r&lt;r*&gt; python train_pg_f18.py HalfCheetah-v2 -ep 150 –discount 0.95 -n 100 -e 3-l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg –nn_baseline –exp_name hc_b&lt;b*&gt;_r&lt; r*&gt;</td>

  </tr>

 </tbody>

</table>

The run with reward-to-go and the baseline should achieve an average score close to 200. Provide a single plot plotting the learning curves for all four runs.

NOTE: In an earlier version of the homework (before 9/13/18) we had the discount as

0.9, in which case the run with reward-to-go and baseline should achieve an average

score close to 150, and there would not be much difference between the runs with reward-to-go (with or without baseline). If you have already done this part with a discount of 0.9, you do not need to redo the problem, but you would then expect the best scoring run to be about 150 rather than 200.

<strong>8      Bonus!</strong>

Choose any (or all) of the following:

<ul>

 <li>A serious bottleneck in the learning, for more complex environments, is the sample collection time. In py, we only collect trajectories in a single thread, but this process can be fully parallelized across threads to get a useful speedup. Implement the parallelization and report on the difference in training time.</li>

 <li>Implement GAE-<em>λ </em>for advantage estimation.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> Run experiments in a MuJoCo gym environment to explore whether this speeds up training. (Walker2d-v1 may be good for this.)</li>

 <li>In PG, we collect a batch of data, estimate a single gradient, and then discard the data and move on. Can we potentially accelerate PG by taking multiple gradient descent steps with the same batch of data? Explore this option and report on your results. Set up a fair comparison between single-step PG and multi-step PG on at least one MuJoCo gym environment.</li>

</ul>

<h1>9      Submission</h1>

Your report should be a document containing

<ul>

 <li>Your mathematical response (written in L<sup>A</sup>TEX) for Problem 1.</li>

 <li>All graphs requested in Problems 4, 5, 7, and 8.</li>

 <li>Answers to short explanation questions in section 5 and 7.</li>

 <li>All command-line expressions you used to run your experiments.</li>

 <li>(Optionally) Your bonus results (command-line expressions, graphs, and a few sentences that comment on your findings).</li>

</ul>

Please also submit your modified train_pg_f18.py file. If your code includes additional files, provide a zip file including your train_pg_f18.py and all other files needed to run your code. Please include a README.md with instructions needed to exactly duplicate your results (including command-line expressions).

Turn in your assignment by September 19th 11:59pm on Gradescope. Uploade the zip file with your code to <strong>HW2 Code</strong>, and upload the PDF of your report to <strong>HW2</strong>.

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://arxiv.org/abs/1506.02438">https://arxiv.org/abs/1506.02438</a>