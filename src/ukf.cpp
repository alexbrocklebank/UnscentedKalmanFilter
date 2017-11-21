#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

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

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;

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

  // predicted sigma points matrix
  MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Radar measurement dimensions for Update
  n_z_ = 3;	

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // initial state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1.0, 0.0, 0.0, 0.0, 0.0,
	  0.0, 1.0, 0.0, 0.0, 0.0,
	  0.0, 0.0, 1.0, 0.0, 0.0,
	  0.0, 0.0, 0.0, 1.0, 0.0,
	  0.0, 0.0, 0.0, 0.0, 1.0;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Number of Sigma Points
  n_sigpts_ = 1 + (2 * n_x_);
  n_augsigpts_ = 1 + (2 * n_aug_);

  // Weights of sigma points
  weights_ = VectorXd(n_augsigpts_);
  weights_[0] = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);
  for (int i = 1; i < n_augsigpts_; i++)
  {
	  weights_[i] = weight;
  }

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
		x_ = VectorXd(n_x_);
		x_.fill(0.0);

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			// RADAR Initialization
			const double rho = meas_package.raw_measurements_[0];
			double phi = meas_package.raw_measurements_[1];
			const double rhodot = meas_package.raw_measurements_[2];

			// Normalize Angles between -PI/PI
			tools.NormalizeAngle(phi);

			// Derive State Variables from Sensor data
			const double px = rho * cos(phi);
			const double py = rho * sin(phi);
			//const double vx = rhodot * cos(phi);
			//const double vy = rhodot * sin(phi);
			//const double v  = sqrt(vx * vx + vy * vy);
			
			// State Initializes better with zeros
			x_ << px, py, 0.0, 0.0, 0.0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			// LASER Initialization
			const double px = meas_package.raw_measurements_[0];
			const double py = meas_package.raw_measurements_[1];

			// Initialize State with Sensor data and zeros
			x_ << px, py, 0.0, 0.0, 0.0;
		}
		previous_timestamp_ = meas_package.timestamp_;

		is_initialized_ = true;
		return;
	}

	/************************************************************************
	*  PREDICTION
	************************************************************************/

	// Determine elapsed time since last update
	const double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = meas_package.timestamp_;

	// Call Prediction() function
	Prediction(dt);

	/************************************************************************
	*  UPDATE
	************************************************************************/

	// Call corresponding Update function based on sensor data retrieved
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		// RADAR Update
		UpdateRadar(meas_package);
	}
	else {
		// LASER Update
		UpdateLidar(meas_package);
	}
	// Output State Vector and Covariance Matrix for testing
	//cout << "x_ = " << x_ << endl;
	//cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

	// Lesson 7.17: Augmentation

	// Create augmented mean state vector
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.fill(0.0);
	x_aug.head(n_x_) = x_;

	// Create augmented covariance matrix from P_
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;
	// Square root matrix of P_aug
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	MatrixXd Xsig_aug = MatrixXd(n_aug_, n_augsigpts_);
	Xsig_aug.col(0) = x_aug;
	for (int i = 1; i <= n_aug_; i++) {
		Xsig_aug.col(i) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i - 1);
		Xsig_aug.col(i + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i - 1);
	}

	// Lesson 7.20: Sigma Point Prediction
	// Predict sigma points
	Xsig_pred_ = MatrixXd(n_x_, n_augsigpts_);
	Xsig_pred_.fill(0.0);

	for (int i = 0; i < Xsig_aug.cols(); i++)
	{
		// Define the 7 Xk State variables for prediction
		double Px, Py, upsilon, psi, psi_dot, upsilon_a, upsilon_psi_dd = 0.0;
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
		if (psi_dot >= 0.001)
		{
			Px_pred = Px + (up_pd * (sin(p_pddt) - sin(psi))) + (half_dt2 * cos(psi) * upsilon_a);
			Py_pred = Py + (up_pd * (-cos(p_pddt) + cos(psi))) + (half_dt2 * sin(psi) * upsilon_a);
		}
		else
		{
			Px_pred = Px + (upsilon * cos(psi) * delta_t) + (half_dt2 * cos(psi) * upsilon_a);
			Py_pred = Py + (upsilon * sin(psi) * delta_t) + (half_dt2 * sin(psi) * upsilon_a);
		}
		// Shared equations that need not worry about /0
		upsilon_pred = upsilon + (delta_t * upsilon_a);
		psi_pred = psi + (psi_dot * delta_t) + (half_dt2 * upsilon_psi_dd);
		psi_dot_pred = psi_dot + (delta_t * upsilon_psi_dd);

		// Write predicted sigma points into right column
		Xsig_pred_.col(i) << Px_pred, Py_pred, upsilon_pred, psi_pred, psi_dot_pred;
	}

	// Lesson 7.23: Predicted Mean and Covariance
    // The State Mean vector
	VectorXd x = VectorXd(n_x_);
	x.fill(0.0);
    // Predict state mean
    for (int i = 0; i < n_augsigpts_; i++)
    {
        x += weights_[i] * Xsig_pred_.col(i);
    }

	// Prediction covariance matrix
	MatrixXd P = MatrixXd(n_x_, n_x_);
	P.fill(0.0);
    // Predicted State covariance matrix
    for (int i = 0; i < weights_.size(); i++)
    {
        VectorXd difference = Xsig_pred_.col(i) - x;
        // Normalize angle
        tools.NormalizeAngle(difference[3]);
        // Predict state covariance matrix
        P += weights_[i] * (difference * difference.transpose());
    }

	// Store to respective class member
	x_ = x;
	P_ = P;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
	// Measurement noise matrix for lidar
	MatrixXd R_ = MatrixXd(2, 2);
	R_ << std_laspx_, 0, 
		  0, std_laspy_;

	// Lidar covariance matrix
	MatrixXd H_ = MatrixXd(2, 5);
	H_ << 1, 0, 0, 0, 0,
		  0, 1, 0, 0, 0;

	// Vanilla Kalman Filter equations for Lidar
	VectorXd z_pred = H_ * x_;
	VectorXd z = meas_package.raw_measurements_;
	VectorXd y = z - z_pred;

	// Plain KF for Lidar to increase responsiveness
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	// New estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
    // Lesson 7.26: Predict Radar Measurement
	MatrixXd Zsig = MatrixXd(n_z_, n_augsigpts_);
	VectorXd z_pred = VectorXd(n_z_);
	MatrixXd S = MatrixXd(n_z_, n_z_);
	Zsig.fill(0.0);
	z_pred.fill(0.0);
	S.fill(0.0);

	for (int i = 0; i < n_augsigpts_; i++)
	{
		// Variables to store current Sigma Point prediction State Vector
		double Px, Py, upsilon, psi, psi_dot = 0.0;
		Px = Xsig_pred_(0, i);
		Py = Xsig_pred_(1, i);
		upsilon = Xsig_pred_(2, i);
		psi = Xsig_pred_(3, i);
		//psi_dot = Xsig_pred_(4, i);

		// Transform sigma points into measurement space
		double rho = pow((pow(Px, 2) + pow(Py, 2)), 0.5);
		double phi = atan2(Py, Px);
		// Normalize
		tools.NormalizeAngle(phi);
		// Avoid divide by 0
		if (rho < 0.001)
		{
			rho = 0.001;
		}
		double rho_dot = ((Px * cos(psi) * upsilon) + (Py * sin(psi) * upsilon)) / rho;

		Zsig.col(i) << rho, phi, rho_dot;

		// Calculate mean predicted measurement
		z_pred += weights_[i] * Zsig.col(i);
	}
	// Calculate measurement covariance matrix S
	for (int i = 0; i < Zsig.cols(); i++)
	{
		VectorXd difference = Zsig.col(i) - z_pred;
		tools.NormalizeAngle(difference[1]);
		S += weights_[i] * (difference * difference.transpose());
	}
	// Add R/E to S
	MatrixXd E = MatrixXd(n_z_, n_z_);
	E.fill(0.0);
	E << pow(std_radr_, 2), 0, 0,
		0, pow(std_radphi_, 2), 0,
		0, 0, pow(std_radrd_, 2);
	S += E;

	// Lesson 7.29: UKF Update
	MatrixXd Tc = MatrixXd(n_x_, n_z_);
	Tc.fill(0.0);

	for (int i = 0; i < n_augsigpts_; i++)
	{
		//calculate cross correlation matrix, Tc
		VectorXd X_diff = Xsig_pred_.col(i) - x_;
		tools.NormalizeAngle(X_diff[3]);
		VectorXd Z_diff = Zsig.col(i) - z_pred;
		tools.NormalizeAngle(Z_diff[1]);
		Tc += weights_[i] * X_diff * Z_diff.transpose();
	}

	//calculate Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	VectorXd z = meas_package.raw_measurements_;

	//update state mean and covariance matrix, x & P
	VectorXd z_diff = z - z_pred;
	tools.NormalizeAngle(z_diff[1]);
	x_ += K * z_diff;
	P_ += -K * S * K.transpose();

	//NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
