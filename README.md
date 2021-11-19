# Transatlantic AI Hackathon

This repository contains resources and code for the Transatlantic AI Hackathon.

![][banner]

## Submission details
(see [submission page][submission])

HealtAIness will help users to exercise and train for health and fitness.

Therefore our solution guides users through exercises and tracks changes in physical health and fitness using DepthAI and OpenVino models. It provides reports on the health statuses for doctors, therapists and caregivers by analyzing users’ ability to fulfill exercises.

Our team will build a proof of concept which can be extended to help users safely learn new skills including new dance, yoga postures, physiotherapy and give posture recommendations.

**Planned Work:**

We are planning to use models from Open Model Zoo to get the proof of concept ready during the hackathon. Using the Luxonis hardware we will retrieve and process the posture information from a user and analyse it depending on the exercise. The prototype should allow users to select and be guided through an exercise according to their physical plan.

### Running the Project

```
set FLASK_APP=__init__.py
python -m flask run
```

## Contributing

### Repository structure

```
  scripts     > Reuseable demonstrations of planned features
  resources   > Imagery and promotional material
  application > Our (soon) functional prototype
```

We agreed to use branches (e.g. `feature/[name]` or `fix/[bugname]`) to commit modifications to the prototype.
If you have scripted demos or resources, just push them directly to the main branch.

  [banner]: https://s3-eu-west-1.amazonaws.com/ultrahack-upload/0659e5ecd8c980d4c8004f097a7de465
  [submission]: https://ultrahack.org/health-and-safety-monitoring-deephack/submissions/154ac34f-ccb7-4c05-8df5-b2044ca5c956
