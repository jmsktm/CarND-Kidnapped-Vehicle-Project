/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  double std_x, std_y, std_theta;  // Standard deviations for x, y, and theta

  // TODO: Set standard deviations for x, y, and theta
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);
  
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = x + 1;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double std_x, std_y, std_theta;  // Standard deviations for x, y, and theta

  // TODO: Set standard deviations for x, y, and theta
  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];
  
  std::default_random_engine gen;
  std::normal_distribution<double> noise_x(0, std_x);
  std::normal_distribution<double> noise_y(0, std_y);
  std::normal_distribution<double> noise_theta(0, std_theta);
  
  yaw_rate = (yaw_rate > 0.001) ? yaw_rate : 0.001;
  
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    double factor = velocity / yaw_rate;
    double theta_eff = p.theta + yaw_rate * delta_t;
    p.x += factor * (sin(theta_eff) - sin(p.theta)) + noise_x(gen);
    p.y += factor * (cos(p.theta) - cos(theta_eff)) + noise_y(gen);
    p.theta = theta_eff + noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); i++) {
    LandmarkObs obs = observations[i];
  	double min_dist = INFINITY;
    double min_pred_id = -1;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      LandmarkObs pred = predicted[j];
      
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (distance < min_dist) {
        min_dist = distance;
        min_pred_id = pred.id;
      }
    }
    observations[i].id = min_pred_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // calculate normalization term
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
  
  for (int i = 0; i < num_particles; i++) {
    
    //transform observations to MAP co-ordinate system
    std::vector<LandmarkObs> transf_observations = observations;
    
    Particle particle = particles[i];
    double xp = particle.x;
    double yp = particle.y;
    double theta = particle.theta;
    
    for (unsigned int j = 0; j < transf_observations.size(); j++) {
      LandmarkObs observation = transf_observations[j];
      double xc = observation.x;
      double yc = observation.y;
      
      // Using Homogeneous Tranformation
      double xm = xp + cos(theta) * xc - sin(theta) * yc;
      double ym = yp + sin(theta) * xc + cos(theta) * yc;
      
      transf_observations[j].x = xm;
      transf_observations[j].y = ym;
    }
    
    // Constructing an array of landmarks that are within sensor range.
    vector<LandmarkObs> landmarks;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s single_landmark = map_landmarks.landmark_list[i];
      
      double distance = dist(single_landmark.x_f, single_landmark.y_f, xp, yp);
      if (distance <= sensor_range) {
        LandmarkObs landmark;
        landmark.id = single_landmark.id_i;
        landmark.x = single_landmark.x_f;
        landmark.y = single_landmark.y_f;
        
        landmarks.push_back(landmark);
      }
    }
    
    dataAssociation(landmarks, transf_observations);
    
    double weight = 1.0;
    for (unsigned int j = 0; j < transf_observations.size(); j++) {
      LandmarkObs observation = transf_observations[j];
      double mu_x, mu_y = 0.0;
      double x_obs = observation.x;
      double y_obs = observation.y;
      
      for (unsigned int k = 0; k < landmarks.size(); k++) {
        LandmarkObs landmark = landmarks[k];
        if (observation.id == landmark.id) {
          mu_x = landmark.x;
          mu_y = landmark.y;
        }
      }
      
      // Calculate multivariate Gaussian Probability Distribution Function
      double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
      double prob = gauss_norm * exp(-exponent);
      
      if (prob > 0.0) {
        weight *= prob;
      }
    }
    weights.push_back(weight);
    particles[i].weight = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}