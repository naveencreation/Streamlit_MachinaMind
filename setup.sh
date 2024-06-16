#!/bin/bash

# Update package list and install Java
sudo apt-get update
sudo apt-get install -y openjdk-11-jre-headless

# Set JAVA_HOME environment variable
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Verify Java installation
java -version
