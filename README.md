# Reliable and Efficient Model-based Evaluation for Language Models

This repository implements the paper Reliable and Efficient Model-based Evaluation for Language Models. In the synthetic experiment section, three simulations show the effectiveness of using item response theory (IRT) in language model evaluation. To replicate our experiment, first, you need to set up the environment:
```bash
conda create -n certify python=3.10 -y
conda activate certify
pip install -r requirements.txt
conda env create -f R.yml
```

## Synthetic Experiments
In this section, we demonstrate three benefits of our approach over classical model evaluation: test-independent model comparison, reliable model comparison under test contamination, and efficient model evaluation via computerized adaptive test (CAT). 

### Test-independent model comparison
Classical comparison of different models typically requires all models to be evaluated on an identical dataset. We first show that, even if there is only a slight change in the test set, classical evaluation can lead to misleading comparisons, while IRT can correctly compare models. In the simulation, we construct two test sets and two test takers. The difficult test is given to the smart test taker, and the easy test is given to the dumb test taker. We show that the two test takers will get the same average score through classical evaluation, but we can recover their true ability through IRT. 

To run the experiment with the default parameters:
```bash
python test_dependent_simulation.py --theta_1 1 --theta_2 2 --question_num 1000 --Y_bar 0.7
```
where `theta_1` is the ability of the dumb test taker, `theta_2` is the ability of the smart test taker, `question_num` is the size of the two test sets, `Y_bar` is the parameter to construct the two datasets.

### Reliable model comparison under test contamination
In machine learning, it is very common that some test sets are memorized by some model, and thus the model can get a high rank in the benchmark. In this case, we say the test set is contaminated, and the model is cheating. It is impossible for classical evaluation to detect cheating. But using the cheat IRT model from [Using Deterministic, Gated Item Response Theory Model to detect test cheating due to item compromise](https://pubmed.ncbi.nlm.nih.gov/25106396/), we show that cheating can be detected and both the true ability and cheat ability can be recovered.
To run the experiment with the default parameters:
```bash
python cheat.py --question_num 1000 --contamination_percent 0.8 --testtaker_true_theta 0 --testtaker_cheat_gain 1.5
```
where `question_num` is the number of questions, `contamination_percent` is the percent of the test set that is possible to leak and has been memorized by the model, and `1-contamination_percent` of the test is impossible to leak. The test taker's true ability is `testtaker_true_theta`, and they cheat with `testtaker_cheat_gain`. For our defult parameter, `testtaker_true_theta` is 0.5, and testtaker_cheat_theta is `testtaker_true_theta` + `testtaker_cheat_gain`, i.e. 0 + 1.5  = 1.5. 

### Efficient model evaluation via computerized adaptive test
Classical evaluation requires reviewing all the test questions to evaluate the test taker’s ability. We show that computerized adaptive tests can recover the true ability of the test taker by asking just a fraction of the total question bank. We implement 4 algorithms in this simulation: 
- `--algo random` is random testing, i.e. randomly choose the next item
- `--algo owen` is adaptive testing and chooses the next item whose difficulty parameter is most similar to the estimated theta from the last iteration
- `--algo fisher` is adaptive testing and choose the next item with maximum fisher information
- `--algo efisher` is adaptive testing that chooses the next item with the maximum expected fisher information among all samples of MCMC.

To run the experiment with the default parameters:
```bash
python src/CAT.py --algo fisher --question_num 10000 --subset_question_num 50 --warmup 0 --true_theta 0
```
`question_num` is the size of question bank, `subset_question_num` is the size of questions in the test. `warmup` is the number of random testing questions before we use an adaptive testing algorithm, which can save the converging time, `true_theta` is the true ability of the new test taker. 

To visualize the result:
```bash
python src/CAT_plot.py
```
