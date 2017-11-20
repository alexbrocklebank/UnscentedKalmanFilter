#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO: Initialize these

  Hint: one or more values initialized above might be wildly off...
  */
  // predicted sigma points matrix
  MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Weights of sigma points
  VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	
	/************************************************************************
	*  INITIALIZATION
	************************************************************************/
	if (!is_initialized_) {
		// Shared Initialization

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			// RADAR Initialization

		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			// LASER Initialization

		}

		is_initialized_ = true;
		return;
	}

	/************************************************************************
	*  PREDICTION
	************************************************************************/
	// Determine elapsed time since last update
	const double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = meas_package.timestamp_;

	Prediction(dt);
	/************************************************************************
	*  UPDATE
	************************************************************************/
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		// RADAR Update
		UpdateRadar(meas_package);
	}
	else {
		// LASER Update
		UpdateLidar(meas_package);
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
    // Lesson 7.14: Generating Sigma Points
	Xsig.col(0) = x;
	for (int i = 1; i <= n_x; i++) {
		VectorXd col = VectorXd(n_x);
		//std::cout << "A = " << A.block(0, i-1, n_x, 1);
		col = x + sqrt(lambda + n_x) * A.block(0, i - 1, n_x, 1);
		Xsig.col(i) = col;
	}
	for (int i = 1; i <= n_x; i++) {
		VectorXd col = VectorXd(n_x);
		col = x - sqrt(lambda + n_x) * A.block(0, i - 1, n_x, 1);
		Xsig.col(i + 5) = col;
	}

	// Lesson 7.17: Augmentation
	//create augmented mean state
	x_aug.fill(0.0);
	x_aug.head(5) = x;


	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P;
	P_aug(5, 5) = std_a * std_a;
	P_aug(6, 6) = std_yawdd * std_yawdd;

	//create square root matrix
	MatrixXd A = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 1; i <= n_aug; i++) {
		Xsig_aug.col(i) = x_aug + sqrt(lambda + n_aug) * A.col(i - 1);
		Xsig_aug.col(i + n_aug) = x_aug - sqrt(lambda + n_aug) * A.col(i - 1);
	}

	// Lesson 7.20: Sigma Point Prediction
	// Predict sigma points
	for (int i = 0; i < Xsig_aug.cols(); i++)
	{
		// Define the 5 Xk State variables for prediction
		double Px, Py, upsilon, psi, psi_dot, psi_dot_dot, upsilon_a, upsilon_psi_dd = 0.0;
		Px = Xsig_aug(0, i);
		Py = Xsig_aug(1, i);
		upsilon = Xsig_aug(2, i);
		psi = Xsig_aug(3, i);
		psi_dot = Xsig_aug(4, i);
		upsilon_a = Xsig_aug(5, i);
		upsilon_psi_dd = Xsig_aug(6, i);

		// Define reuseable mathematical segments
		double up_pd = upsilon / psi_dot;
		double p_pddt = psi + (psi_dot * delta_t);
		double half_dt2 = 0.5 * pow(delta_t, 2);

		// Initialize variables to store prediction
		double Px_pred, Py_pred, upsilon_pred, psi_pred, psi_dot_pred = 0.0;
		// Avoid division by zero in prediction Xk+1 State equations
		if (psi_dot >= 0.0001)
		{
			Px_pred = Px + (up_pd * (sin(p_pddt) - sin(psi))) + (half_dt2 * cos(psi) * upsilon_a);
			Py_pred = Py + (up_pd * (-cos(p_pddt) + cos(psi))) + (half_dt2 * sin(psi) * upsilon_a);
			upsilon_pred = upsilon + (delta_t * upsilon_a);
			psi_pred = psi + (psi_dot * delta_t) + (half_dt2 * upsilon_psi_dd);
			psi_dot_pred = psi_dot + (delta_t * upsilon_psi_dd);
		}
		else
		{
			Px_pred = Px + (upsilon * cos(psi) * delta_t) + (half_dt2 * cos(psi) * upsilon_a);
			Py_pred = Py + (upsilon * sin(psi) * delta_t) + (half_dt2 * sin(psi) * upsilon_a);
			upsilon_pred = upsilon + (delta_t * upsilon_a);
			psi_pred = psi + (psi_dot * delta_t) + (half_dt2 * upsilon_psi_dd);
			psi_dot_pred = psi_dot + (delta_t * upsilon_psi_dd);
		}
		// Write predicted sigma points into right column
		Xsig_pred.col(i) << Px_pred, Py_pred, upsilon_pred, psi_pred, psi_dot_pred;

		// Lesson 7.23: Predicted Mean and Covariance
		// Initialize vectors/matrices to zeros for safe sum operations
		weights.fill(0.0);
		x.fill(0.0);
		P.fill(0.0);

		// Set weights
		for (int i = 0; i < weights.size(); i++)
		{
			// -32.003396  -31.910729
			if (i == 0)
			{
				weights[i] = lambda / (lambda + n_aug);
			}
			else
			{
				weights[i] = 0.5 / (lambda + n_aug);
			}
			// Predict state mean
			for (int j = 0; j < x.size(); j++)
			{
				x[j] += weights[i] * Xsig_pred(j, i);
			}
		}
		for (int i = 0; i < weights.size(); i++)
		{
			VectorXd difference = Xsig_pred.col(i) - x;
			// Normalize angle
			difference[3] = atan2(sin(difference[3]), cos(difference[3]));
			// Predict state covariance matrix
			P += weights[i] * (difference * difference.transpose());
		}
	}

	// Lesson 7.26: Predict Radar Measurement
	Zsig.fill(0.0);
	z_pred.fill(0.0);
	S.fill(0.0);

	for (int i = 0; i < Xsig_pred.cols(); i++)
	{
		// Variables to store current Sigma Point prediction State Vector
		double Px, Py, upsilon, psi, psi_dot = 0.0;
		Px = Xsig_pred(0, i);
		Py = Xsig_pred(1, i);
		upsilon = Xsig_pred(2, i);
		psi = Xsig_pred(3, i);
		psi_dot = Xsig_pred(4, i);

		// Transform sigma points into measurement space
		double rho = pow((pow(Px, 2) + pow(Py, 2)), 0.5);
		double phi = atan((Py / Px));
		// Normalize
		phi = atan2(sin(phi), cos(phi));
		double rho_dot = ((Px * cos(psi) * upsilon) + (Py * sin(psi) * upsilon)) / rho;


		Zsig.col(i) << rho, phi, rho_dot;

		for (int j = 0; j < n_z; j++)
		{
			// Calculate mean predicted measurement
			z_pred[j] += weights[i] * Zsig(j, i);
		}
	}
	// Calculate measurement covariance matrix S
	for (int i = 0; i < Zsig.cols(); i++)
	{
		VectorXd difference = Zsig.col(i) - z_pred;
		difference[1] = atan2(sin(difference[1]), cos(difference[1]));
		S += weights[i] * (difference * difference.transpose());
	}
	// Add R/E to S
	MatrixXd E = MatrixXd(3, 3);
	E.fill(0.0);
	E << pow(std_radr, 2), 0, 0,
		0, pow(std_radphi, 2), 0,
		0, 0, pow(std_radrd, 2);
	S += E;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
	// Lesson 7.29: UKF Update
	Tc.fill(0.0);

	for (int i = 0; i < Xsig_pred.cols(); i++)
	{
		//calculate cross correlation matrix, Tc
		VectorXd X_diff = Xsig_pred.col(i) - x;
		X_diff[3] = atan2(sin(X_diff[3]), cos(X_diff[3]));
		VectorXd Z_diff = Zsig.col(i) - z_pred;
		Z_diff[1] = atan2(sin(Z_diff[1]), cos(Z_diff[1]));
		Tc += weights[i] * X_diff * Z_diff.transpose();
	}

	//calculate Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//update state mean and covariance matrix, x & P
	VectorXd z_diff = z - z_pred;
	z_diff[1] = atan2(sin(z_diff[1]), cos(z_diff[1]));
	x += K * z_diff;
	P += -K * S * K.transpose();
}
