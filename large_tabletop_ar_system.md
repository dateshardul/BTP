# 10m x 6m Tabletop AR Defense Training System
## Complete Technical Specification for Large-Scale Indoor AR Maps

## System Overview

**Target Configuration:**
- **Table Size**: 10 meters × 6 meters (conference room/briefing room scale)
- **Map Scale**: Configurable (1:1000 to 1:10000 typical for tactical maps)
- **Users**: 8-20 simultaneous personnel around the table
- **Viewing**: 360° access - users can walk around entire perimeter
- **Height**: Standard table height (~75-80cm) with AR map floating 5-10cm above surface

## Key Technical Challenges for Large Tables

### **Challenge 1: QR Code Visibility**
- **Problem**: Single QR code cannot be seen from all positions around a 10m×6m table
- **Solution**: Multiple QR code strategy with coordinate hierarchy

### **Challenge 2: Tracking at Scale**
- **Problem**: AR SLAM may drift over 10m distances
- **Solution**: Multi-anchor reference system with continuous validation

### **Challenge 3: Multi-User Synchronization**
- **Problem**: 20 users need perfect spatial alignment
- **Solution**: Distributed anchor sharing with sub-centimeter precision

## QR Code Anchor Strategy

### **Multi-QR Code Layout for 10m×6m Table**

```
                    10 meters
    ┌─────────────────────────────────────────┐
    │ [QR1]                           [QR2]   │ 6
    │                                         │
    │                                         │ m
    │           TACTICAL MAP AREA             │ e
    │                                         │ t
    │                                         │ e
    │ [QR3]                           [QR4]   │ r
    │                                         │ s
    └─────────────────────────────────────────┘

QR Code Positions:
- QR1: Corner (0,0) - Master Anchor
- QR2: Corner (10,0) - Secondary Anchor  
- QR3: Corner (0,6) - Validation Anchor
- QR4: Corner (10,6) - Validation Anchor

Optional Additional Anchors:
- QR5: Center (5,3) - Primary Reference
- QR6: Mid-edges as needed
```

### **QR Code Implementation Architecture**

```csharp
public class LargeTableAnchorSystem : MonoBehaviour
{
    [Header("Table Configuration")]
    public Vector2 tableSize = new Vector2(10.0f, 6.0f);
    public float tableHeight = 0.75f;
    public float mapHoverHeight = 0.08f; // 8cm above table
    
    [Header("QR Code Configuration")]
    public float qrCodeSize = 0.15f; // 15cm QR codes for better detection range
    public int requiredAnchorsForInitialization = 2;
    public int maxAnchorsForValidation = 4;
    
    [Header("Multi-User Support")]
    public int maxSimultaneousUsers = 20;
    public float spatialSyncTolerance = 0.02f; // 2cm tolerance
    
    private Dictionary<string, QRCodeAnchor> detectedAnchors = new Dictionary<string, QRCodeAnchor>();
    private TableCoordinateSystem coordinateSystem;
    private ARAnchor masterTableAnchor;
    private GameObject tacticalMapInstance;
    private bool isTableInitialized = false;
    
    private void Start()
    {
        InitializeMultiQRDetection();
    }
    
    private void InitializeMultiQRDetection()
    {
        var qrDetector = GetComponent<QRCodeDetector>();
        qrDetector.OnQRCodeDetected += HandleQRCodeDetection;
        qrDetector.OnQRCodeLost += HandleQRCodeLost;
        
        // Configure for large table detection
        qrDetector.detectionRange = 8.0f; // Extended range for large table
        qrDetector.minQRSize = 0.10f; // Minimum 10cm for distant detection
        qrDetector.maxQRSize = 0.20f; // Maximum 20cm for close detection
        
        qrDetector.StartContinuousDetection();
    }
    
    private void HandleQRCodeDetection(QRCodeData qrData)
    {
        // Validate QR code belongs to this table
        if (!ValidateTableQRCode(qrData))
        {
            return;
        }
        
        // Add or update anchor
        var anchorId = qrData.anchorId;
        var qrAnchor = new QRCodeAnchor
        {
            id = anchorId,
            position = qrData.transform.position,
            rotation = qrData.transform.rotation,
            tablePosition = qrData.encodedTablePosition,
            confidence = qrData.detectionConfidence,
            lastSeen = DateTime.Now
        };
        
        detectedAnchors[anchorId] = qrAnchor;
        
        // Check if we can initialize the table coordinate system
        if (!isTableInitialized && detectedAnchors.Count >= requiredAnchorsForInitialization)
        {
            AttemptTableInitialization();
        }
        else if (isTableInitialized)
        {
            // Validate and refine existing coordinate system
            ValidateCoordinateSystem(qrAnchor);
        }
    }
    
    private void AttemptTableInitialization()
    {
        // Find the best pair of anchors to establish coordinate system
        var anchorPair = FindBestAnchorPair();
        
        if (anchorPair != null)
        {
            // Create coordinate system
            coordinateSystem = CreateTableCoordinateSystem(anchorPair.anchor1, anchorPair.anchor2);
            
            // Create master table anchor
            masterTableAnchor = CreateMasterTableAnchor();
            
            // Initialize tactical map
            InitializeTacticalMap();
            
            isTableInitialized = true;
            
            Debug.Log($"Table initialized with anchors: {anchorPair.anchor1.id}, {anchorPair.anchor2.id}");
        }
    }
    
    private AnchorPair FindBestAnchorPair()
    {
        float bestDistance = 0f;
        AnchorPair bestPair = null;
        
        var anchors = detectedAnchors.Values.ToArray();
        
        // Find the pair with the largest distance (corner-to-corner is ideal)
        for (int i = 0; i < anchors.Length; i++)
        {
            for (int j = i + 1; j < anchors.Length; j++)
            {
                var distance = Vector3.Distance(anchors[i].position, anchors[j].position);
                
                // Prefer corner-to-corner pairs
                bool isCornerPair = IsCornerPair(anchors[i], anchors[j]);
                
                if (distance > bestDistance || (isCornerPair && distance > bestDistance * 0.8f))
                {
                    bestDistance = distance;
                    bestPair = new AnchorPair { anchor1 = anchors[i], anchor2 = anchors[j] };
                }
            }
        }
        
        return bestPair;
    }
    
    private TableCoordinateSystem CreateTableCoordinateSystem(QRCodeAnchor anchor1, QRCodeAnchor anchor2)
    {
        // Calculate table coordinate system from two known anchors
        var tableOrigin = CalculateTableOrigin(anchor1, anchor2);
        var tableRotation = CalculateTableRotation(anchor1, anchor2);
        
        return new TableCoordinateSystem
        {
            origin = tableOrigin,
            rotation = tableRotation,
            scale = Vector3.one,
            tableSize = tableSize,
            anchor1 = anchor1,
            anchor2 = anchor2,
            confidence = (anchor1.confidence + anchor2.confidence) / 2f
        };
    }
    
    private Vector3 CalculateTableOrigin(QRCodeAnchor anchor1, QRCodeAnchor anchor2)
    {
        // Use encoded table positions to determine where table origin should be
        var tablePos1 = anchor1.tablePosition; // e.g., (0,0) for corner
        var tablePos2 = anchor2.tablePosition; // e.g., (10,6) for opposite corner
        
        var worldPos1 = anchor1.position;
        var worldPos2 = anchor2.position;
        
        // Calculate table origin in world coordinates
        // This assumes QR codes encode their position on the table (0,0) to (10,6)
        var tableDirection = (worldPos2 - worldPos1).normalized;
        var tableDistance = Vector3.Distance(worldPos1, worldPos2);
        
        // Calculate where (0,0) would be in world space
        var expectedTableDistance = Vector3.Distance(
            new Vector3(tablePos1.x, 0, tablePos1.y),
            new Vector3(tablePos2.x, 0, tablePos2.y)
        );
        
        var scale = tableDistance / expectedTableDistance;
        
        // Find world position of table origin (0,0)
        var originOffset = new Vector3(-tablePos1.x * scale, 0, -tablePos1.y * scale);
        var worldOrigin = worldPos1 + originOffset;
        
        return worldOrigin;
    }
    
    private void InitializeTacticalMap()
    {
        if (tacticalMapInstance != null)
        {
            DestroyImmediate(tacticalMapInstance);
        }
        
        // Load tactical map prefab/texture
        var mapPrefab = LoadTacticalMapPrefab();
        
        // Instantiate map at table position
        tacticalMapInstance = Instantiate(mapPrefab);
        tacticalMapInstance.transform.SetParent(masterTableAnchor.transform);
        tacticalMapInstance.transform.localPosition = Vector3.up * mapHoverHeight;
        tacticalMapInstance.transform.localRotation = Quaternion.identity;
        tacticalMapInstance.transform.localScale = new Vector3(tableSize.x, 1f, tableSize.y);
        
        // Configure map interaction
        SetupMapInteraction();
        
        // Enable multi-user synchronization
        EnableMultiUserSync();
    }
    
    private void ValidateCoordinateSystem(QRCodeAnchor newAnchor)
    {
        if (coordinateSystem == null) return;
        
        // Calculate where this anchor should be based on current coordinate system
        var expectedWorldPos = coordinateSystem.TableToWorldPosition(newAnchor.tablePosition);
        var actualWorldPos = newAnchor.position;
        
        var error = Vector3.Distance(expectedWorldPos, actualWorldPos);
        
        if (error > spatialSyncTolerance)
        {
            Debug.LogWarning($"Anchor {newAnchor.id} position error: {error:F3}m");
            
            // Attempt coordinate system refinement
            RefineCoordinateSystem();
        }
    }
    
    private void RefineCoordinateSystem()
    {
        // Use all available anchors to improve coordinate system accuracy
        var validAnchors = detectedAnchors.Values
            .Where(a => DateTime.Now - a.lastSeen < TimeSpan.FromSeconds(5))
            .ToArray();
            
        if (validAnchors.Length >= 3)
        {
            // Use least squares fit to refine coordinate system
            coordinateSystem = OptimizeCoordinateSystem(validAnchors);
            
            // Update master anchor position
            UpdateMasterAnchor();
        }
    }
}

// Supporting Data Structures
public class QRCodeAnchor
{
    public string id;
    public Vector3 position;
    public Quaternion rotation;
    public Vector2 tablePosition; // Position on table (0,0) to (10,6)
    public float confidence;
    public DateTime lastSeen;
}

public class TableCoordinateSystem
{
    public Vector3 origin;
    public Quaternion rotation;
    public Vector3 scale;
    public Vector2 tableSize;
    public QRCodeAnchor anchor1;
    public QRCodeAnchor anchor2;
    public float confidence;
    
    public Vector3 TableToWorldPosition(Vector2 tablePos)
    {
        // Convert table coordinates (0,0)-(10,6) to world position
        var localPos = new Vector3(tablePos.x, 0, tablePos.y);
        return origin + (rotation * Vector3.Scale(localPos, scale));
    }
    
    public Vector2 WorldToTablePosition(Vector3 worldPos)
    {
        // Convert world position back to table coordinates
        var localPos = Quaternion.Inverse(rotation) * (worldPos - origin);
        localPos = Vector3.Scale(localPos, new Vector3(1f/scale.x, 1f/scale.y, 1f/scale.z));
        return new Vector2(localPos.x, localPos.z);
    }
}

public struct AnchorPair
{
    public QRCodeAnchor anchor1;
    public QRCodeAnchor anchor2;
}
```

### **QR Code Encoding Schema for Large Tables**

```json
{
  "version": "1.0",
  "anchorId": "TABLE_ALPHA_QR1",
  "tableId": "BRIEFING_ROOM_ALPHA",
  "anchorType": "CORNER_MASTER",
  "tablePosition": {"x": 0.0, "y": 0.0},
  "tableSize": {"width": 10.0, "height": 6.0},
  "mapConfiguration": {
    "mapId": "OPERATION_THUNDER_TACTICAL",
    "scale": "1:5000",
    "classification": "SECRET",
    "lastUpdated": "2024-01-15T10:30:00Z"
  },
  "anchorMetadata": {
    "description": "Master corner anchor - bottom left",
    "installationDate": "2024-01-10",
    "qrCodeSize": 0.15
  }
}
```

### **Multi-User Synchronization for Large Tables**

```csharp
public class LargeTableMultiUserSync : MonoBehaviour
{
    [Header("Synchronization")]
    public float syncUpdateRate = 30f; // 30 Hz
    public float maxSyncDistance = 0.05f; // 5cm max desync
    
    private Dictionary<string, UserAnchorState> userStates = new Dictionary<string, UserAnchorState>();
    private NetworkManager networkManager;
    
    private void Start()
    {
        networkManager = GetComponent<NetworkManager>();
        networkManager.OnUserJoined += HandleUserJoined;
        networkManager.OnUserPositionUpdate += HandleUserPositionUpdate;
        
        InvokeRepeating(nameof(BroadcastAnchorState), 0f, 1f / syncUpdateRate);
    }
    
    private void HandleUserJoined(string userId, DeviceInfo deviceInfo)
    {
        // Send current table anchor state to new user
        var anchorState = new TableAnchorState
        {
            coordinateSystem = coordinateSystem,
            detectedAnchors = detectedAnchors,
            mapConfiguration = GetCurrentMapConfiguration(),
            timestamp = DateTime.Now
        };
        
        networkManager.SendToUser(userId, "table_anchor_state", anchorState);
    }
    
    private void BroadcastAnchorState()
    {
        if (!isTableInitialized) return;
        
        // Send lightweight anchor updates to all users
        var update = new AnchorUpdate
        {
            masterAnchorPosition = masterTableAnchor.transform.position,
            masterAnchorRotation = masterTableAnchor.transform.rotation,
            confidence = coordinateSystem.confidence,
            timestamp = DateTime.Now
        };
        
        networkManager.BroadcastToAll("anchor_update", update);
    }
    
    private void HandleUserPositionUpdate(string userId, Vector3 position, Quaternion rotation)
    {
        // Track user positions relative to table
        var tablePosition = coordinateSystem.WorldToTablePosition(position);
        
        userStates[userId] = new UserAnchorState
        {
            userId = userId,
            worldPosition = position,
            worldRotation = rotation,
            tablePosition = tablePosition,
            timestamp = DateTime.Now
        };
        
        // Check for anchor drift across users
        CheckForAnchorDrift(userId);
    }
    
    private void CheckForAnchorDrift(string userId)
    {
        // Compare this user's anchor understanding with others
        var userState = userStates[userId];
        var driftDetected = false;
        var maxDrift = 0f;
        
        foreach (var otherUser in userStates.Values)
        {
            if (otherUser.userId == userId) continue;
            
            var drift = Vector3.Distance(userState.worldPosition, otherUser.worldPosition);
            if (drift > maxSyncDistance)
            {
                driftDetected = true;
                maxDrift = Mathf.Max(maxDrift, drift);
            }
        }
        
        if (driftDetected)
        {
            Debug.LogWarning($"Anchor drift detected for user {userId}: {maxDrift:F3}m");
            
            // Initiate anchor re-synchronization
            RequestAnchorResync(userId);
        }
    }
}
```

## Key Technical Specifications

### **Performance Targets**
- **QR Detection Range**: 3-8 meters (depending on QR size)
- **Spatial Accuracy**: <2cm across entire 10m×6m surface
- **Update Rate**: 30Hz for anchor synchronization
- **User Capacity**: 20 simultaneous users
- **Initialization Time**: <10 seconds with 2+ QR codes visible

### **Hardware Requirements**
- **QR Code Size**: 15cm × 15cm (optimal for large table detection)
- **QR Code Placement**: Minimum 4 corners, optional center/edges
- **Table Height**: 75-80cm standard conference table
- **Room Lighting**: 200-1000 lux (typical conference room)

### **Software Architecture**
- **Multi-QR Detection**: Simultaneous tracking of 4-6 QR codes
- **Coordinate System**: Least-squares optimization for accuracy
- **SLAM Integration**: ARFoundation + custom anchor management
- **Network Sync**: Real-time multi-user coordinate sharing

## Next Implementation Steps

### **Phase 1: Basic Multi-QR System**
1. Implement multi-QR code detection
2. Create coordinate system calculation
3. Test with 2-4 QR codes on large table

### **Phase 2: SLAM Integration**
1. Integrate ARFoundation SLAM tracking
2. Implement anchor persistence
3. Test spatial accuracy across 10m distance

### **Phase 3: Multi-User Synchronization**
1. Implement network synchronization
2. Add drift detection and correction
3. Test with multiple AR devices

**Which aspect would you like me to detail further?** The QR code detection system, coordinate system mathematics, or multi-user synchronization architecture?