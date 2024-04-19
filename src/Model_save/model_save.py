# """
# Saving model for deployment
# """
#
# import os
# from joblib import dump
# from sklearn.ensemble import RandomForestClassifier
#
#
# def save_model(model: RandomForestClassifier, dir_path: str) -> None:
#     try:
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#         dump(model, os.path.join(dir_path, 'obesity_classifier.joblib'))
#         print('Model saved successfully')
#     except Exception as e:
#         print(f'Error while saving model: {e}')
