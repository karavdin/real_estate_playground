# Review-based User Recommendations
The goal of this project is build a recommendation service for users on a vacation rental platform based on their previous experience.

# Idea of the solution

We use the k-means algorithm to cluster all the listings based on the reviews. The features would form the unsupervised clusters based on TF-IDF scores of the text.

**How do we do that ?**

Each listings's reviews are collected and concatenated as a single string. Thus, each listings has the feature set of tf-idf scores for the concatenated string of reviews. Further the tf-idf scores as a feature set is used to find the euclidean distance between selected points in space, thus allowing us to implement the k-means algorithm.

**What is TF-IDF score ?**

Given a **document**(concatenated string of a listing) in a **corpus**(across the reviews of all listings), It tells how rarely a word occurs accross the corpus and how frequently it occurs in a that particular document.

**Example for intution**

Consider comparing reviews of chocolates. Let's assume there are three variants in chocolates available in the market. 

***Review for Variant 1*** : This is the best choclate in the world.

***Review for Variant 2*** : I liked this choclate.

Given that similarity of two sentences here is based on Euclidean distance, the reviews would have closer distance due the presence of the word " Chocolate". 

However, there would a be lot of noice and misallocations, but it's possibility is very less as the reviews for rental places would involve some amount of context to express the thoughts. Also we concatenate all the reviews for the listing-users pairs and enchance it with the listing description, which alltogether should reduce the noise by considering the tf-idf scores for each word.


# Highlights of Exploratory data analysis
- Most of reviews are written in English, however ~22% are written in non-english languages
- Listings are presented in 33 neighbourhoods in London
- 73% of listings have review
- Most of the users write just 1 review, but some write much more. The maximum number of reviews per user is 60
- On average a listing is reviewed by 73 users. Some listing are much more popular and get up to 600 reviews


# Scope definition
English is the mostly used language and thus I focused only on reviews written in English. Some locations don't have any reviews (most probably because they only recently introduced on the platform). For simplicity they are ignored and only the listing with reviews are used in the modeling and recommendations.

# Modeling 
As discussed above clustering based on TF-IDF score is used. This is a very time and resources consuming algorithm, therefore number of clusters were restricted to 60 with maximum 240 features for TF-IDF score.
The model provides a list of locations from the user cluster, which sometimes might contain hundreds of listings. Three randomly sampled listings are shown to user.

# Example of recommendation
Let's check recommendation for the user, who wrote following review for one of his previous stays:
"The location is a 5-minute walk to the borough station, which is very convenient.\nThere are also 3 bedrooms, which are very good value for money.\nThe response from the landlord was also quick and very helpful"
From this review we can notice that user appreciate when host is quick and responsive and that the place is close to metro station.
Our model suggested following listings for this user: [31325432, 20098244, 35024701].
Here are reviews other users wrote for them:
** 31325432 **
'I had a great stay!'\n
'Great location near Wembley and frequent trains to the city'
** 20098244 **
'Great host \nGood communication and great place to stay' \n
'Great stay, very close to Gunnersbury. Large, comfortable beds and very clean. Ricardo was a great host.'
** 35024701 ** 
'You can expect kindness, and a cute ambient',
'Great place 15 minutes walk from tube. Quiet at night and Samantha is great crack',
'Sam is very hospitable.',
'Samantha was a very gracious host who went out of her way to be helpful. I highly recommend her accommodation!'

From the reviews we can conclude that all suggested locations have responsive host and also located close to metro station. Based on facts we know about the user, most probably (s)he will find these lisings worth to check too.


# Outlook
During one-day project it was not possible to cover many things. Below is list of open items, which could be covered next:
- Evaluation. Here one could start with classification metrics, e.g Precision@k (fraction of top k recommended items that are relevant to the user), Recall@k (fraction of top k recommended items that are in a set of items relevant to the user). However as for any other task, metrics should be selected based on business objective.
- Include listings without reviews
- Introduce ranking of the suggestions for ensure a user gets the most valuable suggestions
