# perceptrons
Implementation of simple perceptron API, single layer neural-network (Adaline: ADAptive LInear NEuron), and stochastic gradient descent.

# Artificial Neuron (Single Perceptron)

## Biological Neuron
- A neuron is an interconnected nerve cell in the brain 
- Multiple signals arrive at the dendrites (connection to the neuron), they are integrated into the cell body (inputs), and if an accumulated signal from all inputs exceeds a certain threshold, an output signal is generated 
## Formal Definition
$$w = \begin{vmatrix} w_1 \\ ... \\ w_m \end{vmatrix}, x = \begin{vmatrix} x_1 \\ ... \\ x_m \end{vmatrix}$$ 
- In the context of binary classification with two classes (0 and 1), we have the following values: 
	- $\sigma(z)$ is the **decision function** that takes a linear combination of certain input values with corresponding weight
	- $x$ represents the input values
	- $w$ represents their corresponding weights 
	- $z$ is the so-called net input $z = w_1x_1 + w_2x_2 + ... + w_mx_m$ (dot product) 
- Similarly to the description of the biological neuron, if the net input of a particular example $x^i$ exceeds a defined threshold $\theta$, we predict class $1$ in the binary classification. Else $0$ 
- More formally this decision function is a variation of a **unit step function**
	- $\sigma (z) = \begin{cases} 1: z \ge \theta \\ 0: \text{ otherwise} \end{cases}$ 
### What does it mean for this to be a "reductionist" approach? 
- Either a neuron *fires* or it doesn't 
## Redefining the Decision Function 
- We define a *bias unit* $b = -\theta$ 
- If $z \ge \theta \rightarrow z - \theta \ge 0$
- Since $z = w^Tx$, we can say that $z =w^Tx + b$ 
- Now, the decision function can be redefined as: 
	- $\sigma (z) = \begin{cases} 1: z \ge 0 \\ 0: \text{ otherwise} \end{cases}$
## Linear Optimization Objective Function 
- This is represented as an objective function $z = w^Tx$ 
![[Pasted image 20250412184513.png]] 
## Perceptron Learning Rule
1. Initialize the weights and bias unit to 0 or small random numbers 
2. For each training example $x^i$: 
	1. Compute the output value $\hat{y^i}$
	2. Update the weights and bias unit 
## The Perceptron Learning Rule Defined Formally 
1. The output value is the class label predicted by the unit step function defined earlier. The update of the bias unit and each weight can be defined as follows: 
	1. $\begin{cases} w_j:=w_j + \Delta w_j \\ b:=b + \Delta b\end{cases}$ where "deltas" are computed as $\begin{cases}\Delta w_j = \eta (y^i - \hat{y^i})x_j^i \\ \Delta b = \eta (y^i - \hat{y^i})\end{cases}$  
		1. $\eta$ represents the "learning rate" which controls how much the weights and biases are adjusted during each step (typically a value between $0$ and $1$)
		2. $\hat{y^i}$ is the predicted label
		3. $y^i$ is the true label of the $i^{th}$ training example (therefore the difference is the error)
	2. It's important to note that the each $w^i$ corresponds to a feature $x^i$. Therefore, the feature influences the $\Delta w_j$ value, but not the $\Delta b$ value 
### Thought Experiment to Illustrate the Learning Rule 
- If the perceptron successfully predicts the class label, the bias and weight units remain unchanged since the error $y^i - \hat{y^i} = 0$
- But if the prediction is incorrect, the weights are pushed toward the direction of the correct target class
	- $y^i = 1, \hat{y^i} = 0, \Delta w_j = \eta (1-0)x_j^i = \eta x_j^i, \Delta b = \eta (1-0) = \eta$ 
	- $y^i = 0, \hat{y^i} = 1, \Delta w_j = \eta (0-1)x_j^i = -\eta x_j^i, \Delta b = \eta (0-1) = -\eta$ 
### Under what circumstances can the perceptron converge? 
- If the two classes are linearly separable, the two classes can converge. That is, if they can be perfectly separated by a linear decision boundary. 
![[Pasted image 20250412190958.png]] 
- Note that if they can't be separated, we can set a maximum number of passes over the training dataset (epochs) (i.e. a maximum amount of tolerated misclassifications)
