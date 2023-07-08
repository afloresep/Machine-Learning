## PART 2: BAYES THEOREM II
Let’s continue to try to classify the review “This crib was amazing”.

The second part of Bayes’ Theorem is a bit more extensive. We now want to compute 
P(positive | review) = P (review | positive) · P(positive)
                            P(review)

In other words, if we assume that the review is positive, what is the probability that the words “This”, “crib”, “was”, and “amazing” are the only words in the review?

To find this, we have to assume that each word is conditionally independent. This means that one word appearing doesn’t affect the probability of another word from showing up. This is a pretty big assumption!

We now have this equation. You can scroll to the right to see the full equation.

P(“This crib was amazing" ∣ positive)=P(“This" ∣ positive)⋅P(“crib" ∣ positive)...

Let’s break this down even further by looking at one of these terms. P("crib"|positive) is the probability that the word “crib” appears in a positive review. To find this, we need to count up the total number of times “crib” appeared in our dataset of positive reviews. If we take that number and divide it by the total number of words in our positive review dataset, we will end up with the probability of “crib” appearing in a positive review

P ("crib" | positive) =  # of “crib" in positive / # of words in positive

Let’s first find the total number of words in all positive reviews and store that number in a variable named total_pos.

To do this, we can use the built-in Python sum() function. sum() takes a list as a parameter. The list that you want to sum is the values of the dictionary pos_counter, which you can get by using pos_counter.values().

Do the same for total_neg.
Create two variables named pos_probability and neg_probability. Each of these variables should start at 1. These are the variables we are going to use to keep track of the probabilities.
Create a list of the words in review and store it in a variable named review_words. You can do this by using Python’s .split() function.
For example if the string test contained "Hello there", then test.split() would return ["Hello", "there"].​
Loop through every word in review_words. Find the number of times word appears in pos_counter and neg_counter. Store those values in variables named word_in_pos and word_in_neg.

In the next steps, we’ll use this variable inside the for loop to do a series of multiplications.
Inside the for loop, set pos_probability to be pos_probability multiplied by word_in_pos / total_pos.

This step is finding each term to be multiplied together. For example, when word is "crib", you’re calculating the following:
**P ("crib" | positive) =  # of “crib" in positive / # of words in positive**
Do the same multiplication for neg_probability.
Outside the for loop, print both pos_probability and neg_probability. Those values are P(“This crib was amazing”|positive) and P(“This crib was amazing”|negative).


## SMOOTHING
In the last exercise, one of the probabilities that we computed was the following:

P(“crib" ∣ positive)=  # of words in positive / # of “crib" in positive
​But what happens if “crib” was never in any of the positive reviews in our dataset? This fraction would then be 0, and since everything is multiplied together, the entire probability P(review | positive) would become 0.

This is especially problematic if there are typos in the review we are trying to classify. If the unclassified review has a typo in it, it is very unlikely that that same exact typo will be in the dataset, and the entire probability will be 0. To solve this problem, we will use a technique called smoothing.

In this case, we smooth by adding 1 to the numerator of each probability and N to the denominator of each probability.**N is the number of unique words in our review dataset**

For example, P("crib" | positive) goes from this:
P(“crib" ∣ positive)= # of words in positive / # of “crib" in positive
​
 
To this:

P(“crib" ∣ positive)=  # of words in positive+N / # of “crib" in positive+1
​

In the denominator of those fractions, add the number of unique words in the appropriate dataset.
For the positive probability, this should be the length of pos_counter which can be found using len().
Again, make sure to put parentheses around your denominator so the division happens after the addition!


## Classify 

If we look back to Bayes’ Theorem, we’ve now completed both parts of the numerator. We now need to multiply them together
P(positive ∣ review)= *P(review | positive)⋅P(positive)* / P(review)

Let’s now consider the denominator P(review). In our small example, this is the probability that “This”, “crib”, “was”, and “amazing” are the only words in the review. Notice that this is extremely similar to P(review | positive). The only difference is that we don’t assume that the review is positive.

However, before we start to compute the denominator, let’s think about what our ultimate question is. We want to predict whether the review “This crib was amazing” is a positive or negative review. In other words, we’re asking whether P(positive | review) is greater than P(negative | review). If we expand those two probabilities, we end up with the following equations.

P(positive ∣ review)= P(review | positive)⋅P(positive) / P(review)
P(negative ∣ review)= P(review | negative)⋅P(negative) / P(review)​

Notice that P(review) is in the denominator of each. That value will be the same in both cases! Since we’re only interested in comparing these two probabilities, there’s no reason why we need to divide them by the same value. We can completely ignore the denominator!

Let’s see if our review was more likely to be positive or negative!

INSTRUCTIONS
After the for loop, multiply pos_probability by percent_pos and neg_probability by percent_neg. Store the two values in final_pos and final_neg and print both.

Compare final_pos to final_neg:

If final_pos was greater than final_neg, print "The review is positive".
Otherwise print "The review is negative".

