using UnityEngine;
using Unity.Netcode;
using Meta.XR.MRUtilityKit;

/// <summary>
/// BACKUP SCRIPT for table edge clipping
/// Primary method: Meta Quest 3's automatic MR occlusion (recommended)
/// This script: Manual fallback if automatic occlusion doesn't work
///
/// Features:
/// - Detects table boundaries from Scene API
/// - Visual warning when terrain extends beyond table
/// - Optional automatic constraint to keep terrain on table
/// - Debug visualizer for table edges
/// </summary>
public class TerrainBoundsManager : NetworkBehaviour
{
    [Header("References")]
    [SerializeField] private Transform terrainTransform;
    [SerializeField] private SurfaceAnchorManager surfaceAnchor;

    [Header("Bounds Detection")]
    [SerializeField] private bool autoDetectTableBounds = true;
    [SerializeField] private float boundsPadding = 0.1f;  // Keep terrain 10cm inside edges

    [Header("Constraint Mode")]
    [SerializeField] private BoundsMode boundsMode = BoundsMode.WarningOnly;
    [SerializeField] private bool showBoundsVisualizer = true;

    [Header("Visual Feedback")]
    [SerializeField] private Color inBoundsColor = Color.green;
    [SerializeField] private Color outOfBoundsColor = Color.red;
    [SerializeField] private LineRenderer boundsVisualizer;

    public enum BoundsMode
    {
        Disabled,           // No bounds checking
        WarningOnly,        // Show warning but allow overflow
        SoftConstraint,     // Gentle push back toward bounds
        HardConstraint      // Prevent any movement beyond bounds
    }

    // Table bounds data
    private MRUKAnchor tableSurface;
    private Bounds tableBounds;
    private bool boundsValid = false;
    private bool terrainExceedsBounds = false;

    // Network-synced warning state
    private NetworkVariable<bool> showingBoundsWarning = new NetworkVariable<bool>(
        false,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        if (autoDetectTableBounds)
        {
            // Wait a bit for MR room data to be available
            Invoke(nameof(DetectTableBounds), 1f);
        }

        // Create visualizer if needed
        if (showBoundsVisualizer && boundsVisualizer == null)
        {
            CreateBoundsVisualizer();
        }

        // Subscribe to warning changes
        showingBoundsWarning.OnValueChanged += OnBoundsWarningChanged;
    }

    public override void OnNetworkDespawn()
    {
        if (showingBoundsWarning != null)
        {
            showingBoundsWarning.OnValueChanged -= OnBoundsWarningChanged;
        }
        base.OnNetworkDespawn();
    }

    private void Update()
    {
        if (boundsMode == BoundsMode.Disabled || !boundsValid) return;

        // Check if terrain is within bounds
        CheckTerrainBounds();

        // Apply constraints if needed
        if (terrainExceedsBounds)
        {
            HandleBoundsViolation();
        }
    }

    /// <summary>
    /// Detect table bounds from MR Utility Kit
    /// </summary>
    public void DetectTableBounds()
    {
        if (surfaceAnchor == null)
        {
            Debug.LogWarning("SurfaceAnchorManager not assigned. Cannot detect table bounds.");
            return;
        }

        // Get table surface from anchor manager
        tableSurface = surfaceAnchor.SelectedSurface;

        if (tableSurface == null)
        {
            Debug.LogWarning("No table surface detected yet. Retrying...");
            Invoke(nameof(DetectTableBounds), 2f);  // Retry in 2 seconds
            return;
        }

        // Get table dimensions from MR Utility Kit
        if (tableSurface.PlaneRect.HasValue)
        {
            Rect planeRect = tableSurface.PlaneRect.Value;

            // Calculate bounds with padding
            Vector3 center = tableSurface.transform.position;
            Vector3 size = new Vector3(
                planeRect.width - (boundsPadding * 2),
                1.0f,  // Allow some vertical room
                planeRect.height - (boundsPadding * 2)
            );

            tableBounds = new Bounds(center, size);
            boundsValid = true;

            Debug.Log($"Table bounds detected - Size: {planeRect.width:F2}m x {planeRect.height:F2}m");

            // Update visualizer
            UpdateBoundsVisualizer();
        }
        else
        {
            Debug.LogWarning("Table surface has no dimension data");
        }
    }

    /// <summary>
    /// Check if terrain is currently within table bounds
    /// </summary>
    private void CheckTerrainBounds()
    {
        if (terrainTransform == null) return;

        Renderer terrainRenderer = terrainTransform.GetComponent<Renderer>();
        if (terrainRenderer == null) return;

        Bounds terrainWorldBounds = terrainRenderer.bounds;

        // Check if terrain extends beyond table on horizontal plane
        bool xOverflow = terrainWorldBounds.min.x < tableBounds.min.x ||
                        terrainWorldBounds.max.x > tableBounds.max.x;

        bool zOverflow = terrainWorldBounds.min.z < tableBounds.min.z ||
                        terrainWorldBounds.max.z > tableBounds.max.z;

        bool wasExceeding = terrainExceedsBounds;
        terrainExceedsBounds = xOverflow || zOverflow;

        // Update warning state on server
        if (IsServer && terrainExceedsBounds != wasExceeding)
        {
            showingBoundsWarning.Value = terrainExceedsBounds;
        }

        // Update visualizer color
        if (boundsVisualizer != null)
        {
            Color targetColor = terrainExceedsBounds ? outOfBoundsColor : inBoundsColor;
            boundsVisualizer.startColor = targetColor;
            boundsVisualizer.endColor = targetColor;
        }
    }

    /// <summary>
    /// Handle terrain extending beyond table bounds
    /// </summary>
    private void HandleBoundsViolation()
    {
        switch (boundsMode)
        {
            case BoundsMode.WarningOnly:
                // Just show visual warning (handled in CheckTerrainBounds)
                break;

            case BoundsMode.SoftConstraint:
                // Gently push terrain back toward center
                ApplySoftConstraint();
                break;

            case BoundsMode.HardConstraint:
                // Prevent movement beyond bounds
                ApplyHardConstraint();
                break;
        }
    }

    /// <summary>
    /// Gently push terrain back if it goes too far beyond edges
    /// </summary>
    private void ApplySoftConstraint()
    {
        if (terrainTransform == null) return;

        Vector3 currentPos = terrainTransform.position;
        Vector3 constrainedPos = ClampPositionToBounds(currentPos);

        // Lerp toward constrained position (soft/elastic feel)
        Vector3 newPos = Vector3.Lerp(currentPos, constrainedPos, Time.deltaTime * 2f);

        terrainTransform.position = newPos;
    }

    /// <summary>
    /// Hard clamp terrain position to table bounds
    /// </summary>
    private void ApplyHardConstraint()
    {
        if (terrainTransform == null) return;

        Vector3 constrainedPos = ClampPositionToBounds(terrainTransform.position);
        terrainTransform.position = constrainedPos;
    }

    /// <summary>
    /// Clamp position to stay within table bounds
    /// </summary>
    private Vector3 ClampPositionToBounds(Vector3 position)
    {
        if (!boundsValid) return position;

        // Only clamp horizontal position (X and Z), keep Y unchanged
        return new Vector3(
            Mathf.Clamp(position.x, tableBounds.min.x, tableBounds.max.x),
            position.y,
            Mathf.Clamp(position.z, tableBounds.min.z, tableBounds.max.z)
        );
    }

    /// <summary>
    /// Create visual bounds indicator (shows table edges)
    /// </summary>
    private void CreateBoundsVisualizer()
    {
        GameObject visualizerObj = new GameObject("TableBoundsVisualizer");
        visualizerObj.transform.SetParent(transform);

        boundsVisualizer = visualizerObj.AddComponent<LineRenderer>();
        boundsVisualizer.startWidth = 0.02f;
        boundsVisualizer.endWidth = 0.02f;
        boundsVisualizer.loop = true;
        boundsVisualizer.useWorldSpace = true;
        boundsVisualizer.material = new Material(Shader.Find("Sprites/Default"));
        boundsVisualizer.positionCount = 5;  // 4 corners + back to start

        // Will update when bounds are detected
    }

    /// <summary>
    /// Update bounds visualizer to show table edges
    /// </summary>
    private void UpdateBoundsVisualizer()
    {
        if (boundsVisualizer == null || !boundsValid) return;

        Vector3 center = tableBounds.center;
        Vector3 extents = tableBounds.extents;
        float y = center.y + 0.05f;  // Slightly above table

        // Draw rectangle around table edges
        Vector3[] corners = new Vector3[5]
        {
            new Vector3(center.x - extents.x, y, center.z - extents.z),  // Bottom-left
            new Vector3(center.x - extents.x, y, center.z + extents.z),  // Top-left
            new Vector3(center.x + extents.x, y, center.z + extents.z),  // Top-right
            new Vector3(center.x + extents.x, y, center.z - extents.z),  // Bottom-right
            new Vector3(center.x - extents.x, y, center.z - extents.z)   // Back to start
        };

        boundsVisualizer.SetPositions(corners);
        boundsVisualizer.startColor = inBoundsColor;
        boundsVisualizer.endColor = inBoundsColor;
    }

    private void OnBoundsWarningChanged(bool previousValue, bool newValue)
    {
        if (newValue)
        {
            Debug.LogWarning("Terrain extending beyond table edges!");
        }
    }

    /// <summary>
    /// Check if a specific point is within table bounds
    /// </summary>
    public bool IsPointOnTable(Vector3 point)
    {
        if (!boundsValid) return true;
        return tableBounds.Contains(point);
    }

    /// <summary>
    /// Get percentage of terrain that's off the table (0-1)
    /// </summary>
    public float GetOffTablePercentage()
    {
        if (!boundsValid || terrainTransform == null) return 0f;

        Renderer terrainRenderer = terrainTransform.GetComponent<Renderer>();
        if (terrainRenderer == null) return 0f;

        Bounds terrainBounds = terrainRenderer.bounds;

        // Calculate overflow on each side
        float overflowXMin = Mathf.Max(0, tableBounds.min.x - terrainBounds.min.x);
        float overflowXMax = Mathf.Max(0, terrainBounds.max.x - tableBounds.max.x);
        float overflowZMin = Mathf.Max(0, tableBounds.min.z - terrainBounds.min.z);
        float overflowZMax = Mathf.Max(0, terrainBounds.max.z - tableBounds.max.z);

        // Calculate total overflow area vs terrain area
        float totalOverflow = overflowXMin + overflowXMax + overflowZMin + overflowZMax;
        float terrainSize = terrainBounds.size.x + terrainBounds.size.z;

        return Mathf.Clamp01(totalOverflow / terrainSize);
    }

    /// <summary>
    /// Manually set table bounds (if automatic detection fails)
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void SetTableBoundsServerRpc(Vector3 center, Vector3 size)
    {
        if (!IsServer) return;

        tableBounds = new Bounds(center, size);
        boundsValid = true;

        Debug.Log($"Table bounds manually set: {size.x:F2}m x {size.z:F2}m");

        UpdateBoundsVisualizer();
    }

    // Public properties
    public Bounds TableBounds => tableBounds;
    public bool BoundsValid => boundsValid;
    public bool TerrainExceedsBounds => terrainExceedsBounds;
    public float OffTablePercentage => GetOffTablePercentage();
    public BoundsMode CurrentMode => boundsMode;

    // Editor gizmos for visualization
    private void OnDrawGizmosSelected()
    {
        if (!boundsValid) return;

        // Draw table bounds
        Gizmos.color = terrainExceedsBounds ? outOfBoundsColor : inBoundsColor;

        Vector3 center = tableBounds.center;
        Vector3 extents = tableBounds.extents;
        float y = center.y;

        // Draw bottom face of bounds (table surface)
        Vector3[] corners = new Vector3[4]
        {
            center + new Vector3(-extents.x, 0, -extents.z),
            center + new Vector3(-extents.x, 0, extents.z),
            center + new Vector3(extents.x, 0, extents.z),
            center + new Vector3(extents.x, 0, -extents.z)
        };

        // Draw table outline
        Gizmos.DrawLine(corners[0], corners[1]);
        Gizmos.DrawLine(corners[1], corners[2]);
        Gizmos.DrawLine(corners[2], corners[3]);
        Gizmos.DrawLine(corners[3], corners[0]);

        // Draw corner markers
        foreach (var corner in corners)
        {
            Gizmos.DrawWireSphere(corner, 0.05f);
        }

        // Draw center
        Gizmos.DrawWireSphere(center, 0.03f);
    }
}

/*
 * USAGE NOTES:
 *
 * PRIMARY METHOD (Recommended):
 * ===============================
 * Use Meta Quest 3's built-in MR occlusion - NO CODE NEEDED!
 *
 * Setup in Unity:
 * 1. Select OVRCameraRig in Hierarchy
 * 2. Find OVRManager component
 * 3. Quest Features → Enable "Depth Submission"
 * 4. Quest Features → Enable "Scene Support"
 * 5. Done! Parts of terrain beyond table will be automatically hidden
 *
 * BACKUP METHOD (This Script):
 * ===============================
 * Use only if automatic occlusion doesn't work
 *
 * Setup:
 * 1. Add this component to Terrain GameObject
 * 2. Assign SurfaceAnchorManager reference
 * 3. Choose Bounds Mode:
 *    - WarningOnly: Shows red outline when terrain goes off table
 *    - SoftConstraint: Gently pushes terrain back
 *    - HardConstraint: Prevents movement beyond edges
 * 4. Enable "Show Bounds Visualizer" to see table edges
 *
 * The automatic occlusion is MUCH better because:
 * - Uses real depth data from Quest cameras
 * - Works with any surface shape
 * - No performance overhead
 * - More realistic
 */