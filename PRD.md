# Product Requirements Document (PRD)

# CourtVision

**Version:** 1.0 (MVP)
**Owner:** Fawehinmi Erioluwa
**Status:** In Development

---

# 1. Overview

CourtVision is an AI-powered basketball analytics platform that converts ordinary basketball video into structured game intelligence.

Instead of simply detecting players, CourtVision understands what is happening in a game by identifying:

* Players
* Basketball
* Court
* Teams
* Possession
* Passes
* Shots
* Game events
* Player statistics

The long-term goal is to become a full basketball intelligence engine capable of analyzing games from amateur leagues to professional competitions.

---

# 2. Vision

To make advanced basketball analytics accessible to everyone using only video.

Instead of requiring expensive tracking hardware, CourtVision should work from ordinary camera footage.

---

# 3. Problem Statement

Basketball analytics solutions today are expensive because they require:

* Optical tracking systems
* Multiple synchronized cameras
* Wearable sensors
* Dedicated arena hardware

Small clubs, schools, and individual coaches cannot afford these systems.

CourtVision solves this by using computer vision and AI to extract similar insights directly from video.

---

# 4. Goals

## MVP Goals

Detect:

* Players
* Basketball

Track:

* Player identities
* Ball movement

Generate:

* Player movement
* Basic statistics
* Possession estimates
* Pass counts

---

## Future Goals

Automatically detect:

* Shot attempts
* Makes
* Misses
* Rebounds
* Assists
* Turnovers
* Steals
* Blocks
* Fouls
* Screens
* Pick and Roll
* Defensive formations
* Offensive sets

---

# 5. Target Users

Primary:

* Basketball coaches

Secondary:

* Players
* Scouts
* Schools
* Basketball academies

Future:

* Professional teams
* Broadcast companies
* Sports media
* Basketball researchers

---

# 6. Core Features

## Player Detection

Uses YOLOv8 COCO model.

Responsibilities:

* Detect every player
* Draw bounding boxes
* Generate player IDs
* Track players between frames

---

## Ball Detection

Uses a custom fine-tuned YOLO model trained on Roboflow.

Responsibilities:

* Detect basketball
* Track basketball
* Estimate ball trajectory
* Provide confidence scores

---

## Multi Object Tracking

Uses ByteTrack.

Responsibilities:

* Persistent player IDs
* Persistent ball tracking
* Smooth tracking between frames

---

## Team Classification

Determines team membership using jersey colors.

Responsibilities:

* Extract dominant jersey color
* Cluster jersey colors
* Assign Team A or Team B

Future:

* Learn jersey appearance automatically.

---

## Player Statistics

Tracks:

* Distance travelled
* Average speed
* Maximum speed
* Frames visible

Future:

* Fatigue estimation
* Heatmaps
* Sprint count

---

## Possession Tracking

Determines who controls the basketball.

Inputs:

* Ball position
* Player position

Outputs:

Current possession

Example:

Player 7 possesses the ball.

---

## Pass Detection

Detects passes by observing changes in possession.

Example:

Player 3
↓

Player 8

Pass recorded.

---

## Commentary Engine

Generates human-readable game commentary.

Example:

Player 8 receives the ball.

Player 8 passes to Player 12.

Fast break initiated.

---

## Shot Detection

Future feature.

Should detect:

* Shot release
* Ball arc
* Rim interaction
* Made shot
* Missed shot

---

## Match Data Storage

Stores:

Players

Teams

Possessions

Passes

Events

Statistics

---

# 7. AI Models

## Model 1

YOLOv8n

Purpose:

Player detection

Status:

Implemented

---

## Model 2

Custom YOLO

Purpose:

Basketball detection

Status:

Implemented

Training:

Roboflow dataset

Fine-tuned on basketball images

---

## Model 3

Custom Rim Detector

Purpose:

Basketball rim detection

Status:

Planned

---

## Future Models

Shot classifier

Pose estimation

Action recognition

Play recognition

Court segmentation

Player re-identification

---

# 8. System Pipeline

Video

↓

Frame Extraction

↓

Player Detection

↓

Ball Detection

↓

Tracking

↓

Team Classification

↓

Statistics

↓

Possession

↓

Pass Detection

↓

Commentary

↓

Analytics Output

---

# 9. Current Tech Stack

Language

Python

Computer Vision

OpenCV

Object Detection

Ultralytics YOLO

Tracking

ByteTrack

Dataset

Roboflow

Machine Learning

PyTorch

Development

Google Colab

Version Control

Git + GitHub

---

# 10. Success Metrics

MVP should achieve:

Player detection

> 95%

Ball detection

> 90%

Player ID stability

> 90%

Pass detection

> 85%

Possession accuracy

> 85%

---

# 11. Current Development Status

Completed

* Player detection
* Ball detection
* ByteTrack integration
* Player tracking
* Ball tracking
* Team classification
* Player statistics
* Possession tracking
* Pass detection framework
* Commentary engine
* Match data storage

In Progress

* Improve custom ball detector
* Increase ball detection consistency
* Integrate custom model into production pipeline
* Validate analytics outputs

Planned

* Rim detector
* Shot detector
* Rebound detection
* Heatmaps
* Event timeline
* Web dashboard
* API
* Real-time processing

---

# 12. Long-Term Roadmap

## Phase 1 — MVP

* Player detection
* Ball detection
* Tracking
* Statistics
* Possession
* Passes

## Phase 2 — Advanced Analytics

* Shot detection
* Rim detection
* Makes and misses
* Heatmaps
* Event timeline

## Phase 3 — Basketball Intelligence

* Offensive play recognition
* Defensive formations
* Automatic play diagrams
* AI-generated coaching reports
* Advanced player comparison

## Phase 4 — Platform

* Cloud processing
* REST API
* Web dashboard
* Mobile application
* Live game analytics

---

# 13. Product Vision Statement

CourtVision aims to become the most accessible AI-powered basketball analytics platform by transforming ordinary game footage into professional-grade insights without requiring specialized hardware. By combining computer vision, object detection, tracking, and sports intelligence, CourtVision will enable coaches, players, scouts, and teams at every level to make smarter decisions through automated, data-driven analysis.
