COMP 6721 Summer 1 Project

Author(s): 
Supervised: Raghav Senwal 
CNN: Raghav Senwal
Semi-supervised: Alireza Lorestani, Nasim Fani

Trained models have been included in the src file, one can easily load them using joblib library and use the model to verify classification on unseen data.

Code example:
```
model_path = '<insert_external_path>/<insert_model_name>.joblib'
loaded_clf = joblib.load(model_path)
loaded_y_pred = loaded_clf.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_y_pred)
print(f'Loaded Model Accuracy: {loaded_accuracy * 100:.2f}%')
```
Trained model for CNN is uploaded on Moodle.

Code example to load model:
```
model.load_state_dict(torch.load('<model_name>.pth'))

```