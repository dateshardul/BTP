using UnityEngine;
using Unity.Netcode;
using Meta.XR.MRUtilityKit;

/// <summary>
/// Manages terrain anchoring to physical surfaces (tables, floors)
/// Uses Meta Quest's Scene Understanding to detect and anchor to surfaces
/// </summary>
public class SurfaceAnchorManager : NetworkBehaviour
{
    [Header("Terrain Reference")]
    [SerializeField] private Transform terrainTransform;

    [Header("Anchor Settings")]
    [SerializeField] private bool autoAnchorOnStart = false;
    [SerializeField] private float anchorHeight = 0.05f;  // Height above surface (5cm)
    [SerializeField] private bool alignToSurfaceNormal = true;

    [Header("Surface Detection")]
    [SerializeField] private LayerMask surfaceLayer;
    [SerializeField] private float maxRaycastDistance = 5f;

    // Current anchor state
    private NetworkVariable<Vector3> anchoredPosition = new NetworkVariable<Vector3>(
        Vector3.zero,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    private NetworkVariable<Quaternion> anchoredRotation = new NetworkVariable<Quaternion>(
        Quaternion.identity,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    private NetworkVariable<bool> isAnchored = new NetworkVariable<bool>(
        false,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    // MR Utility Kit references
    private MRUKRoom currentRoom;
    private MRUKAnchor selectedSurface;

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Subscribe to anchor changes
        anchoredPosition.OnValueChanged += OnAnchoredPositionChanged;
        anchoredRotation.OnValueChanged += OnAnchoredRotationChanged;
        isAnchored.OnValueChanged += OnAnchorStateChanged;

        if (autoAnchorOnStart && IsServer)
        {
            AttemptAutoAnchor();
        }
    }

    public override void OnNetworkDespawn()
    {
        if (anchoredPosition != null) anchoredPosition.OnValueChanged -= OnAnchoredPositionChanged;
        if (anchoredRotation != null) anchoredRotation.OnValueChanged -= OnAnchoredRotationChanged;
        if (isAnchored != null) isAnchored.OnValueChanged -= OnAnchorStateChanged;

        base.OnNetworkDespawn();
    }

    /// <summary>
    /// Attempt to automatically anchor to nearest horizontal surface
    /// </summary>
    private void AttemptAutoAnchor()
    {
        // Try to get current room from MR Utility Kit
        currentRoom = MRUKRoom.GetCurrentRoom();

        if (currentRoom == null)
        {
            Debug.LogWarning("No MR room detected. Waiting for room data...");
            // Subscribe to room creation event
            MRUKRoom.OnRoomsUpdatedEvent.AddListener(OnRoomsUpdated);
            return;
        }

        FindAndAnchorToSurface();
    }

    private void OnRoomsUpdated()
    {
        currentRoom = MRUKRoom.GetCurrentRoom();
        if (currentRoom != null)
        {
            MRUKRoom.OnRoomsUpdatedEvent.RemoveListener(OnRoomsUpdated);
            FindAndAnchorToSurface();
        }
    }

    /// <summary>
    /// Find best surface and anchor terrain to it
    /// </summary>
    private void FindAndAnchorToSurface()
    {
        if (currentRoom == null) return;

        // Find horizontal surfaces (tables, desks, floors)
        var surfaces = currentRoom.GetAllAnchors(new MRUKAnchor.SceneLabels[]
        {
            MRUKAnchor.SceneLabels.TABLE,
            MRUKAnchor.SceneLabels.COUCH,
            MRUKAnchor.SceneLabels.OTHER,  // Flat surfaces
            MRUKAnchor.SceneLabels.FLOOR
        });

        if (surfaces.Count == 0)
        {
            Debug.LogWarning("No suitable surfaces found for anchoring");
            return;
        }

        // Find closest horizontal surface
        MRUKAnchor closestSurface = null;
        float closestDistance = float.MaxValue;

        Vector3 terrainPos = terrainTransform != null ? terrainTransform.position : Vector3.zero;

        foreach (var surface in surfaces)
        {
            // Check if surface is mostly horizontal
            if (Vector3.Dot(surface.transform.up, Vector3.up) > 0.8f)  // Within ~36 degrees of horizontal
            {
                float distance = Vector3.Distance(surface.transform.position, terrainPos);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestSurface = surface;
                }
            }
        }

        if (closestSurface != null)
        {
            AnchorToSurface(closestSurface);
        }
        else
        {
            Debug.LogWarning("No horizontal surface found");
        }
    }

    /// <summary>
    /// Anchor terrain to specified surface
    /// </summary>
    public void AnchorToSurface(MRUKAnchor surface)
    {
        if (!IsServer)
        {
            Debug.LogWarning("Only server can anchor terrain");
            return;
        }

        selectedSurface = surface;

        // Calculate anchor position (center of surface + height offset)
        Vector3 position = surface.transform.position + Vector3.up * anchorHeight;

        // Calculate rotation (align to surface or keep horizontal)
        Quaternion rotation;
        if (alignToSurfaceNormal)
        {
            rotation = Quaternion.FromToRotation(Vector3.up, surface.transform.up);
        }
        else
        {
            // Keep terrain horizontal regardless of surface angle
            rotation = Quaternion.identity;
        }

        // Set networked values
        anchoredPosition.Value = position;
        anchoredRotation.Value = rotation;
        isAnchored.Value = true;

        Debug.Log($"Terrain anchored to {surface.Label} at {position}");
    }

    /// <summary>
    /// Anchor terrain to a point hit by raycast
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void AnchorToPointServerRpc(Vector3 point, Vector3 normal)
    {
        if (!IsServer) return;

        // Position slightly above hit point
        Vector3 position = point + normal * anchorHeight;

        // Align to surface normal if enabled
        Quaternion rotation;
        if (alignToSurfaceNormal)
        {
            rotation = Quaternion.FromToRotation(Vector3.up, normal);
        }
        else
        {
            rotation = Quaternion.identity;
        }

        anchoredPosition.Value = position;
        anchoredRotation.Value = rotation;
        isAnchored.Value = true;

        Debug.Log($"Terrain anchored to point {position}");
    }

    /// <summary>
    /// Remove anchor and allow free movement
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void RemoveAnchorServerRpc()
    {
        if (!IsServer) return;

        isAnchored.Value = false;
        selectedSurface = null;

        Debug.Log("Terrain anchor removed");
    }

    /// <summary>
    /// Manual anchor by selecting a surface the controller is pointing at
    /// </summary>
    [ServerRpc(RequireOwnership = false)]
    public void AnchorAtControllerRayServerRpc(Vector3 rayOrigin, Vector3 rayDirection)
    {
        if (!IsServer) return;

        Ray ray = new Ray(rayOrigin, rayDirection);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, maxRaycastDistance, surfaceLayer))
        {
            AnchorToPointServerRpc(hit.point, hit.normal);
        }
        else
        {
            Debug.LogWarning("No surface hit by controller ray");
        }
    }

    // Network variable callbacks
    private void OnAnchoredPositionChanged(Vector3 previousValue, Vector3 newValue)
    {
        if (terrainTransform != null && isAnchored.Value)
        {
            terrainTransform.position = newValue;
        }
    }

    private void OnAnchoredRotationChanged(Quaternion previousValue, Quaternion newValue)
    {
        if (terrainTransform != null && isAnchored.Value && alignToSurfaceNormal)
        {
            terrainTransform.rotation = newValue;
        }
    }

    private void OnAnchorStateChanged(bool previousValue, bool newValue)
    {
        Debug.Log($"Anchor state: {(newValue ? "Anchored" : "Free")}");
    }

    /// <summary>
    /// Constrain terrain movement to surface plane
    /// Call this when terrain is being manipulated
    /// </summary>
    public Vector3 ConstrainToSurfacePlane(Vector3 desiredPosition)
    {
        if (!isAnchored.Value) return desiredPosition;

        // Project position onto surface plane
        Vector3 normal = anchoredRotation.Value * Vector3.up;
        Vector3 anchorPoint = anchoredPosition.Value;

        // Calculate projection
        Vector3 toDesired = desiredPosition - anchorPoint;
        Vector3 projected = Vector3.ProjectOnPlane(toDesired, normal);

        return anchorPoint + projected;
    }

    // Public properties
    public bool IsAnchored => isAnchored.Value;
    public Vector3 AnchorPosition => anchoredPosition.Value;
    public Quaternion AnchorRotation => anchoredRotation.Value;
    public MRUKAnchor SelectedSurface => selectedSurface;
}