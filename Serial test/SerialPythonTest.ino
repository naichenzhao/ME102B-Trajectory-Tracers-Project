int i;
int number = 0;
int reading;

void setup() {
  Serial.begin(460800);
  Serial.setTimeout(1);
  pinMode(13, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    reading = Serial.parseInt();  // read from the Serial Monitor
    Serial.println(String(reading));
    Serial.flush();
   }
}
