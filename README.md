# Recommender-System

bhattacharyya - Python implementation of A new similarity measure using Bhattacharyya coefficient for collaborative filtering in sparse data

Many times the dataset used for recommendation is sparse, ex-movielens, netflix. Most of the similarity measures do not perform well on sparse dataset as they compute similarity between two users based on only the co-rated items. The approach proposed in this paper (A new similarity measure using Bhattacharyya coefficient for collaborative filtering in sparse data) overcomes the sparsity problem. It calculates similarity between two users on two levels- local and global. I have implemented it in python.

rs_using_jaya - A hybrid recommender system optimised using Jaya optimisation algorithm

I have developed a hybrid ( content + collaborative)  recommendation system and optimized it using (modified) Jaya optimization. Along with the ratings, the demographic information of the users has also been used to find the similarity between the users. In the movielens dataset the demographic information available are - age, gender, occupation and zipcode. The first three have been used in the proposed hybrid method. The rules used for calculation of difference (similarity) scores are - 
•	Age - for difference of every five years, users have been segregated in different groups.
•	Gender - if the gender is different a difference score of one has been given otherwise the score is 0.
•	Occupation - Same as gender.

The movies used in movielens data have been categorised into 18 genres. Instead of using the ratings directly to calculate similarity between users, the users’ inclination towards particular genre has been used. To calculate a user’s preference for a particular genre an array of size 18 is taken in which all the cells are initialized to 0. After that - 
1.	The average rating given by the user is calculated.
2.	The average rating is subtracted from the current rating of a particular movie and 0.5 is added to it (to avoid making the total zero)
3.	Then the difference in rating is added to the cells corresponding to all the genres to which the movie belongs. 
