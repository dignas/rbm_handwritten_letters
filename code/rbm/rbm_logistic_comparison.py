from rbm_io import RBM_IO
from rbm_augment_dataset import augment_dataset

from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime


if __name__ == "__main__":

	io = RBM_IO()

	X, y = io.load_dataset()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
	X_train, y_train = augment_dataset(X_train, y_train)

	rbm = BernoulliRBM(
		n_components=1024,
		learning_rate=0.02,
		n_iter=50,
		random_state=12345,
		batch_size=64,
		verbose=True,
	)

	classifier_rbm = LogisticRegression(
		max_iter=500,
		verbose=True,
	)
	classifier_standalone_logistic = LogisticRegression(
		max_iter=500,
		verbose=True,
	)
	pipeline = Pipeline(steps=[('rbm', rbm), ('classifier', classifier_rbm)])

	pipeline.fit(X_train, y_train)
	classifier_standalone_logistic.fit(X_train, y_train)

	y_pred = pipeline.predict(X_test)
	y_pred_log = classifier_standalone_logistic.predict(X_test)

	io.save_model(pipeline, f"save_pipeline_{str(datetime.datetime.now().timestamp()).replace('.', '_')}")
	io.save_model(classifier_standalone_logistic, f"save_logistic_{str(datetime.datetime.now().timestamp()).replace('.', '_')}")

	print('RBM + logistic classification result:\n', classification_report(y_test, y_pred))
	print('Logistic classification result:\n', classification_report(y_test, y_pred_log))
