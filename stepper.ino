#include <AccelStepper.h>
#include <ezButton.h>

int input = 0; 
int speed = 0;
long position = 0;
int state = HIGH;
// float position_1 = 4000;

// Define the stepper motor and the pins that is connected to
AccelStepper stepper1(1, 25, 26); // (Type of driver: with 2 pins, STEP, DIR)
ezButton limitSwitch(23);


void setup() {
  
  limitSwitch.setDebounceTime(50);
  pinMode(13, OUTPUT);
  // stepper1.setMaxSpeed(-259);
  // stepper1.setAcceleration(600);
  Serial.begin(460800);
  // zeroing();
  stepper1.setMaxSpeed(13000);
  stepper1.setAcceleration(40000);
  // middle();
  Serial.setTimeout(1);
  // Set maximum speed value for the stepper
}

void loop() {
  limitSwitch.loop();
  state = limitSwitch.getState();
  if(state == LOW){
    position = stepper1.currentPosition() + 200;
  }
    
  // Waits for Serial Input that tells motor to move to certain position
  
  if (Serial.available()) {
    input = Serial.parseInt();
    Serial.flush();

    if (input == 0){
      stepper1.setSpeed(0);
      stepper1.setCurrentPosition(10000); 
      position = 10000;  
   }
    else if (input == 1){
      stepper1.setAcceleration(20000);
      stepper1.setMaxSpeed(18000);
      position = stepper1.currentPosition() - 1500;
   }
    else if (input == 2){
      stepper1.setAcceleration(16000);
      stepper1.setMaxSpeed(16000);
      position = stepper1.currentPosition() - 1000;
   }
    else if (input == 3){
      stepper1.setAcceleration(16000);
      stepper1.setMaxSpeed(16000);
      position = stepper1.currentPosition() + 1000;
   }
    else if (input == 4){
      stepper1.setAcceleration(20000);
      stepper1.setMaxSpeed(18000);
      position = stepper1.currentPosition() + 1500;
   }
    else if (input == 5){
      stepper1.setAcceleration(20000);
      stepper1.setMaxSpeed(22000);
      position = stepper1.currentPosition() + 2000;
   }
    else if (input == 6){
      stepper1.setAcceleration(20000);
      stepper1.setMaxSpeed(22000);
      position = stepper1.currentPosition() - 2000;   
  }
  

  }
  stepper1.moveTo(position);
  stepper1.run();
}
