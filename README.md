# VideoGameRecommendation
Machine Learning project that recommends Video Games based on what the user previously liked

Dataset Citation:
Tamber. (2017). Steam Video Games. Retrieved May 22, 2023,.
https://www.kaggle.com/datasets/tamber/steam-video-games

Built in Python using the "Steam Video Games Dataset", Pandas, and Pytorch.
Uses the playtimes of different games by the same Steam users to predict which games a user may like based on other games they enjoyed.

Includes a training block to train the Neural Network and an output block to use the trained network for recommendations.
Re-training is disabled by default, to re-train the model, set the value of the 'conduct_model_training' variable in 'VideoGameRecommendation.py' to True

Due to the nature of using playtime as the primary learning metric, this project tends to favor games with a longer completion time in its output.
