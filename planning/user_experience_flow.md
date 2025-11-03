## Complete User Experience Flow - First Time Setup

**Last Updated:** 2025-10-22
**Time:** 30-60 seconds per user
**Difficulty:** Extremely intuitive, beginner-friendly

---

## 🎯 User Puts On Headset - What Happens:

### **Screen 1: Welcome (5 seconds)**

```
╔════════════════════════════════════╗
║ Welcome to Multi-User Terrain!     ║
║                                    ║
║ Explore 3D maps together.          ║
║ One teacher controls.              ║
║ Students learn together.           ║
║                                    ║
║          [Start Setup]             ║
║            [Skip]                  ║
╚════════════════════════════════════╝

Step 1 of 7  ▓░░░░░░░
```

**User Action:** Tap "Start Setup"

---

### **Screen 2: Role Selection (10 seconds)**

```
╔════════════════════════════════════╗
║ Choose Your Role                   ║
║                                    ║
║ 👨‍🏫 TEACHER:                       ║
║ • Control the terrain map          ║
║ • Zoom, pan, rotate                ║
║ • Place markers                    ║
║                                    ║
║ 👨‍🎓 STUDENT:                       ║
║ • View the terrain                 ║
║ • See teacher's actions            ║
║ • Learn together                   ║
║                                    ║
║  [I am the Teacher]                ║
║  [I am a Student]                  ║
╚════════════════════════════════════╝

Step 2 of 7  ▓▓░░░░░
```

**User Action:** Tap role button

---

### **Screen 3a: Teacher - Start Session (10 seconds)**

```
╔════════════════════════════════════╗
║ Start Session                      ║
║                                    ║
║ You will host the session.         ║
║ Students will join you.            ║
║                                    ║
║ Make sure you're on Wi-Fi.         ║
║                                    ║
║         [Start as Host]            ║
╚════════════════════════════════════╝

Step 3 of 7  ▓▓▓░░░░
```

**User Action:** Tap "Start as Host"

**What happens:**
- Teacher's Quest becomes server
- Session starts
- Waiting for students...

---

### **Screen 3b: Student - Join Session (10 seconds)**

```
╔════════════════════════════════════╝
║ Join Session                       ║
║                                    ║
║ Join the teacher's session.        ║
║                                    ║
║ Make sure you're on same Wi-Fi     ║
║ as the teacher.                    ║
║                                    ║
║      [Join Teacher's Session]      ║
╚════════════════════════════════════╝

Step 3 of 7  ▓▓▓░░░░
```

**User Action:** Tap "Join Teacher's Session"

**What happens:**
- Searching for teacher's session...
- Connecting...

---

### **Screen 4: Connecting... (2-5 seconds)**

**Teacher sees:**
```
╔════════════════════════════════════╗
║ Waiting for Students...            ║
║                                    ║
║ 📱 Share with students:            ║
║ Network: MyWiFi_5G                 ║
║ Session: Room_A47B                 ║
║                                    ║
║ Students connected: 2              ║
║                                    ║
║ [Continue] (or wait for more)      ║
╚════════════════════════════════════╝
```

**Student sees:**
```
╔════════════════════════════════════╗
║ Connecting to Teacher...           ║
║                                    ║
║ ⏳ Please wait...                  ║
║                                    ║
║ [Animated spinner]                 ║
╚════════════════════════════════════╝

(Auto-advances when connected)
```

---

### **Screen 5: Spatial Setup (10 seconds)**

```
╔════════════════════════════════════╗
║ Spatial Setup                      ║
║                                    ║
║ Same room or remote?               ║
║                                    ║
║ 🏫 SAME ROOM:                      ║
║ • All around one table             ║
║ • Quick 10sec alignment            ║
║                                    ║
║ 🌐 REMOTE:                         ║
║ • Different locations              ║
║ • No alignment needed              ║
║                                    ║
║  [Same Room]  [Remote]             ║
╚════════════════════════════════════╝

Step 5 of 7  ▓▓▓▓░░░
```

**User Action:** Tap based on situation

**If Remote:** Skip to Step 6
**If Same Room:** Continue to Step 5a

---

### **Screen 5a: Alignment - Teacher (10 seconds)**

```
╔════════════════════════════════════╗
║ Teacher: Set Reference             ║
║                                    ║
║ Point controller at:               ║
║ • Classroom door                   ║
║ • Whiteboard edge                  ║
║ • Window frame                     ║
║                                    ║
║ Something everyone can see!        ║
║                                    ║
║ Hold TRIGGER for 1.5 seconds       ║
║                                    ║
║ Ray: ░░░░▓▓▓▓ → ░░▓▓▓▓▓▓           ║
║      White    →    Green            ║
╚════════════════════════════════════╝

(Shows live progress bar as teacher holds trigger)
```

**What teacher sees in VR:**
- Pointer ray from controller
- Ray changes color white → green (progress)
- When green: "Reference set!"

---

### **Screen 5b: Alignment - Student (10 seconds)**

```
╔════════════════════════════════════╗
║ Student: Point at Reference        ║
║                                    ║
║ Point at the SAME object           ║
║ the teacher pointed at.            ║
║                                    ║
║ (Ask teacher: "What did you        ║
║  point at?")                       ║
║                                    ║
║ Hold TRIGGER for 1.5 seconds       ║
║                                    ║
║ Ray: ░░░░▓▓▓▓ → ░░▓▓▓▓▓▓           ║
╚════════════════════════════════════╝

(Auto-advances when complete)
```

**What happens:**
- Student's Quest calculates rotation offset
- Applies correction
- "Alignment complete!"
- Student now sees correct viewpoint

---

### **Screen 6a: Tutorial - Teacher (15 seconds)**

```
╔════════════════════════════════════╗
║ Teacher Controls                   ║
║                                    ║
║ TERRAIN:                           ║
║ • Trigger + Move → Zoom            ║
║ • Trigger + Grip + Move → Pan      ║
║ • Trigger + Stick ←→ → Rotate      ║
║                                    ║
║ MARKERS:                           ║
║ • Button A → Place                 ║
║ • Button B → Undo                  ║
║                                    ║
║ POINTER:                           ║
║ • Just point → Show students       ║
║                                    ║
║ [3D Controller Model Visual]       ║
║                                    ║
║ [I Understand]  [Skip]             ║
╚════════════════════════════════════╝

Step 6 of 7  ▓▓▓▓▓░░
```

**Visual Aid:** Animated 3D controller showing each button/trigger

---

### **Screen 6b: Tutorial - Student (10 seconds)**

```
╔════════════════════════════════════╗
║ Student View                       ║
║                                    ║
║ YOU CAN:                           ║
║ 👁️  Watch teacher's pointer       ║
║    (color shows action)            ║
║                                    ║
║ 👁️  See terrain change            ║
║    (zoom, pan, rotate)             ║
║                                    ║
║ 👁️  See markers appear            ║
║                                    ║
║ 🚶 Walk around table               ║
║   for different angles             ║
║                                    ║
║ You cannot control terrain.        ║
║                                    ║
║      [I'm Ready to Learn!]         ║
╚════════════════════════════════════╝

Step 6 of 7  ▓▓▓▓▓░░
```

---

### **Screen 7: Ready! (5 seconds)**

**Teacher:**
```
╔════════════════════════════════════╗
║ ✅ Ready to Teach!                 ║
║                                    ║
║ The terrain will appear on         ║
║ the table in 3... 2... 1...        ║
║                                    ║
║ Point your controller to show      ║
║ students where to look.            ║
║                                    ║
║ Use controls to teach!             ║
║                                    ║
║           [Start!]                 ║
╚════════════════════════════════════╝

Step 7 of 7  ▓▓▓▓▓▓▓ COMPLETE!
```

**Student:**
```
╔════════════════════════════════════╗
║ ✅ Ready to Learn!                 ║
║                                    ║
║ The terrain will appear on         ║
║ the table shortly.                 ║
║                                    ║
║ Watch for teacher's pointer!       ║
║                                    ║
║ Enjoy the lesson!                  ║
║                                    ║
║           [Start!]                 ║
╚════════════════════════════════════╝

Step 7 of 7  ▓▓▓▓▓▓▓ COMPLETE!
```

**User Action:** Tap "Start!"

**What happens:**
- UI fades out
- Terrain appears on table!
- Brief floating hint: "Point controller to begin" (teacher) or "Watch teacher's pointer" (student)
- Ready to use!

---

## ⏱️ Time Breakdown:

| Step | Teacher | Student | Notes |
|------|---------|---------|-------|
| Welcome | 5s | 5s | Can skip |
| Role Selection | 5s | 5s | Clear choice |
| Connection | 5s | 5s | Auto-connect |
| Waiting | 10s | 5s | Teacher waits, student auto |
| Alignment | +10s | +10s | Only if same room |
| Tutorial | 15s | 10s | Can skip |
| Ready | 5s | 5s | Countdown |
| **Total (Remote)** | **45s** | **35s** | Fast! |
| **Total (Same Room)** | **55s** | **45s** | With alignment |

---

## 🎨 Visual Design Principles:

### **Large, Clear Text:**
- Font size: 48+ points
- High contrast (white on dark)
- Easy to read in VR

### **Big Buttons:**
- Minimum 200x80 pixels
- Easy to tap with controller ray
- Clear labels

### **Progress Indicators:**
- Step counter: "Step 3 of 7"
- Progress bar: Visual completion
- Color coding: White → Green (progress)

### **Minimal Choices:**
- Maximum 2-3 buttons per screen
- Clear, unambiguous options
- No confusing terminology

---

## 🔄 Re-Onboarding:

**If user wants to see tutorial again:**
- Settings menu → "Show Tutorial"
- Or: Hold both grips for 3 seconds
- Tutorial replays
- Can exit anytime

---

## 💡 Smart Features:

### **Auto-Skip for Returning Users:**
- First launch: Full onboarding
- Subsequent launches: Skip straight to app
- Stored in PlayerPrefs

### **Context-Aware Instructions:**
- Different text for teacher vs student
- Different buttons shown based on role
- Adapts to connection state

### **Progress Persistence:**
- If app crashes during onboarding
- Resumes at last completed step
- User doesn't start over

### **Help Always Available:**
- "?" button in corner (always visible)
- Shows current step instructions
- Quick refresher without full tutorial

---

## 🎓 Teacher's Complete Journey:

```
1. Put on Quest → Welcome screen (5s)
2. Tap "Start" → Choose "Teacher" (5s)
3. Tap "Teacher" → "Start as Host" (5s)
4. Tap "Start as Host" → Connecting... (5s)
5. Connected → "Same room or Remote?" (5s)
6. Tap based on situation:
   - Remote → Skip to tutorial
   - Same room → Point at door, hold trigger (10s)
7. Tutorial → See controls, tap "I understand" (15s)
8. Ready screen → Tap "Start!" (5s)
9. ✅ Terrain appears on table!
10. Start teaching!

Total: 45-55 seconds
```

---

## 👨‍🎓 Student's Complete Journey:

```
1. Put on Quest → Welcome screen (5s)
2. Tap "Start" → Choose "Student" (5s)
3. Tap "Student" → "Join Teacher" (5s)
4. Tap "Join" → Connecting... (5s auto)
5. Connected → "Same room or Remote?" (5s)
6. Tap based on situation:
   - Remote → Skip to tutorial
   - Same room → Point at same door, hold trigger (10s)
7. Tutorial → See view tips, tap "Ready!" (10s)
8. Ready screen → Tap "Start!" (5s)
9. ✅ Terrain appears on table!
10. Start learning!

Total: 35-45 seconds
```

---

## 🔥 Edge Cases Handled:

### **What if user skips everything?**
- Defaults: Student role, Remote mode, No alignment
- Still works! Just suboptimal for same-room
- Can reconfigure in settings

### **What if connection fails?**
- Clear error message: "Connection failed. Check Wi-Fi."
- Retry button
- Back button to change settings

### **What if calibration fails?**
- Fallback: "Couldn't detect reference. Try again or skip."
- Skip button → Remote mode (no alignment)
- Retry button

### **What if student joins late?**
- Abbreviated onboarding
- Skips to calibration only
- Joins ongoing session

---

## 🎯 Post-Onboarding UX:

### **Persistent Help (Always Available):**

**Floating "?" Button (Bottom-right corner):**
```
Tap to show:
┌──────────────────────┐
│ Quick Help           │
├──────────────────────┤
│ Controls:            │
│ • Trigger+Move = Zoom│
│ • Button A = Marker  │
│ ...                  │
│                      │
│ [Show Full Tutorial] │
│ [Close]              │
└──────────────────────┘
```

### **Contextual Hints (First 5 minutes):**

**When teacher first holds trigger:**
```
Floating hint (3 seconds):
"Hold and move closer to zoom in!"
```

**When teacher first places marker:**
```
Floating hint:
"Button B to undo! ✓"
```

**After 5 minutes:** Hints stop appearing

---

## ✅ Complete Script List (Now 20!):

**New Alignment & Onboarding (3):**
1. SpatialAlignmentManager.cs
2. ManualAlignmentCalibrator.cs
3. SharedAnchorAlignmentSystem.cs
4. OnboardingManager.cs (4 scripts total for this feature set)

**Previous 17:** (All terrain, networking, teacher controls, etc.)

**Total: 20 production-ready scripts!**

---

**End of User Experience Flow**