// MultiStepper.pde
// -*- mode: C++ -*-
// Use MultiStepper class to manage multiple steppers and make them all move to 
// the same position at the same time for linear 2d (or 3d) motion.

#include <AccelStepper.h>
#include <MultiStepper.h>

// EG X-Y position bed driven by 2 steppers
// Alas its not possible to build an array of these with different pins for each :-(
AccelStepper left_stepper(AccelStepper::FULL4WIRE, 8, 10, 9, 11);
AccelStepper right_stepper(AccelStepper::FULL4WIRE, 4, 6, 5, 7);

const int steps_per_cm = 102;
const float spool_distance = 32;
const float calibration_distance = 16;
float x = 16;
float y = 0;

// Up to 10 steppers can be handled as a group by MultiStepper
MultiStepper steppers;

void setup() {
  Serial.begin(9600);

  // Configure each stepper
  left_stepper.setMaxSpeed(200);
  right_stepper.setMaxSpeed(200);

  // Then give them to MultiStepper to manage
  steppers.addStepper(left_stepper);
  steppers.addStepper(right_stepper);
}

long* calculatePosition(float x, float y) {
  static long result[2];
  float other_x = spool_distance - x;
  result[0] = round((sqrt(x*x + y*y) - calibration_distance) * steps_per_cm);
  result[1] = round(-(sqrt(other_x*other_x + y*y) - calibration_distance) * steps_per_cm);
  return result;
}

void loop() {
  if (Serial.available() > 0){
    x = Serial.parseFloat();
    y = Serial.parseFloat();
    while (Serial.available()) {
      Serial.read();
    }

    steppers.moveTo(calculatePosition(x, y));
    steppers.runSpeedToPosition(); // Blocks until all are in position
    Serial.println((String)"At x=" + x + " y=" + y);
  }
}
