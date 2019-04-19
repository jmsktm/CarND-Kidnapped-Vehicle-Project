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

#define EPSILON 1e-6

using std::string;
using std::vector;
using std::numeric_limits;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (unsigned int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
    weights.push_back(p.weight);
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
  for (unsigned int i = 0; i < num_particles; i++) {
    Particle p = particles[i];

    double new_theta = p.theta + yaw_rate * delta_t;
    double velocity_to_yaw_rate = fabs(yaw_rate) > EPSILON ? velocity / yaw_rate : 0;

    double new_x = p.x + velocity_to_yaw_rate * (sin(new_theta) - sin(p.theta));
    double new_y = p.y + velocity_to_yaw_rate * (cos(p.theta) - cos(new_theta));

    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
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
    LandmarkObs observed_landmark = observations[i];

    int closest_landmark_id = -1;
    double min_dist = numeric_limits<double>::max();

    for (unsigned int j = 0; j < predicted.size(); j++) {
      LandmarkObs predicted_landmark = predicted[j];
      double distance = dist(observed_landmark.x, observed_landmark.y, predicted_landmark.x, predicted_landmark.y);
      if (distance < min_dist) {
        min_dist = distance;
        closest_landmark_id = predicted_landmark.id;
      }
    }
    
    observations[i].id = closest_landmark_id;
  }
}

vector<LandmarkObs> ParticleFilter::getObservationsInMapCoordinates(Particle particle,
                    const vector<LandmarkObs>& observations) {
  vector<LandmarkObs> observations_in_map_coordinates;

  for (unsigned int i = 0; i < observations.size(); i++) {
    LandmarkObs obs = observations[i];
    double x = particle.x + (cos(particle.theta) * obs.x) - (sin(particle.theta) * obs.y);
    double y = particle.y + (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y);
    observations_in_map_coordinates.push_back(LandmarkObs{ obs.id, x, y });
  }

  return observations_in_map_coordinates;
}

vector<LandmarkObs> ParticleFilter::getLandmarksInRange(Particle particle, const Map &map_landmarks,
                                  double sensor_range) {
  vector<LandmarkObs> landmarks_in_range;
  for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++) {
    float landmark_x = map_landmarks.landmark_list[i].x_f;
    float landmark_y = map_landmarks.landmark_list[i].y_f;
    int id = map_landmarks.landmark_list[i].id_i;
    double distance = dist(particle.x, particle.y, landmark_x, landmark_y);
    if (distance < sensor_range) {
      landmarks_in_range.push_back(LandmarkObs{ id, landmark_x, landmark_y });
    }
  }

  return landmarks_in_range;
}

double ParticleFilter::calculateWeight(vector<LandmarkObs> observations, vector<LandmarkObs> landmarks,
                                    double std_landmark[]) {
  double weight = 1.0;

  for (unsigned int i = 0; i < observations.size(); i++) {
    LandmarkObs observation = observations[i];

    vector<LandmarkObs>::iterator closest_landmark = std::find_if(landmarks.begin(), landmarks.end(),
                        [&observation](const LandmarkObs &landmark) {
                          return landmark.id == observation.id;
                        });

    double prob = multiv_prob(std_landmark[0], std_landmark[1], observation.x, observation.y, closest_landmark->x, closest_landmark->y);
    if (prob < EPSILON) {
      weight *= EPSILON;
    } else {
      weight *= prob;
    }
  }

  return weight;
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
  weights.clear();
  for (unsigned int i = 0; i < num_particles; i++) {
    vector<LandmarkObs> observations_map_coord = getObservationsInMapCoordinates(particles[i], observations);
    vector<LandmarkObs> landmarks_in_range = getLandmarksInRange(particles[i], map_landmarks, sensor_range);
    dataAssociation(landmarks_in_range, observations_map_coord);
    particles[i].weight = calculateWeight(observations_map_coord, landmarks_in_range, std_landmark);
    weights.push_back(particles[i].weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	// Define discrete distribution
	std::discrete_distribution<int> d_(weights.begin(), weights.end());

	// Define a resampled particle vector.
	vector<Particle> resampled_vector;

	// Iterate through number of particles and update resampled vector.
	for (int i = 0; i < num_particles; i++) {
		resampled_vector.push_back(particles[d_(gen)]);
	}
	particles = resampled_vector;
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