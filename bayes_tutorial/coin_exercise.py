import coin

# The definition of hypothesis F is "p is 0.5"
# The following code computes L(E|F)

evidence = 140, 110

likelihood_fair = coin.Likelihood(evidence, 0.5) 


# If the definition of hypothesis B is "p is either 0.4 or 0.6 with
# equal probability, find L(E|B).

likelihood_biased = (coin.Likelihood(evidence, 0.4) / 2 + coin.Likelihood(evidence, 0.6) / 2)

print likelihood_fair
print likelihood_biased

print likelihood_biased/(likelihood_biased + likelihood_fair)