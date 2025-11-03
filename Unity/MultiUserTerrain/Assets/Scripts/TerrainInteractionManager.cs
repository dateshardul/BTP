using UnityEngine;
using Unity.Netcode;
using Meta.XR.MRUtilityKit;

public class TerrainInteractionManager : NetworkBehaviour
{
    [Header("Terrain References")]
    [SerializeField] private Transform terrainTransform;
    [SerializeField] private float zoomSpeed = 0.5f;
    [SerializeField] private float panSpeed = 1f;
    [SerializeField] private float rotateSpeed = 50f;
    
    [Header("Interaction Limits")]
    [SerializeField] private Vector2 zoomLimits = new Vector2(0.5f, 3f);
    [SerializeField] private float panRadius = 5f;
    
    // Networked properties - synchronized across all clients (adapted from Fusion [Networked])
    private NetworkVariable<Vector3> networkedPosition = new NetworkVariable<Vector3>(Vector3.zero);
    private NetworkVariable<Quaternion> networkedRotation = new NetworkVariable<Quaternion>(Quaternion.identity);
    private NetworkVariable<float> networkedScale = new NetworkVariable<float>(1f);
    private NetworkVariable<bool> isBeingManipulated = new NetworkVariable<bool>(false);
    private NetworkVariable<ulong> currentManipulatorId = new NetworkVariable<ulong>(0);
    
    // Local interaction state
    private bool isLocallyGrabbed = false;
    private Vector3 lastHandPosition;
    private Vector2 initialPinchDistance;
    
    // Visual feedback
    [SerializeField] private GameObject manipulationIndicator;
    [SerializeField] private Color[] userColors;
    
    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();
        
        // Initialize terrain to networked state (adapted from Fusion Start())
        if (IsServer)
        {
            networkedPosition.Value = terrainTransform.position;
            networkedRotation.Value = terrainTransform.rotation;
            networkedScale.Value = terrainTransform.localScale.x;
        }
        
        // Subscribe to value changes for automatic updates
        networkedPosition.OnValueChanged += OnPositionChanged;
        networkedRotation.OnValueChanged += OnRotationChanged;
        networkedScale.OnValueChanged += OnScaleChanged;
        isBeingManipulated.OnValueChanged += OnManipulationStateChanged;
    }
    
    public override void OnNetworkDespawn()
    {
        // Unsubscribe from value changes
        if (networkedPosition != null) networkedPosition.OnValueChanged -= OnPositionChanged;
        if (networkedRotation != null) networkedRotation.OnValueChanged -= OnRotationChanged;
        if (networkedScale != null) networkedScale.OnValueChanged -= OnScaleChanged;
        if (isBeingManipulated != null) isBeingManipulated.OnValueChanged -= OnManipulationStateChanged;
        
        base.OnNetworkDespawn();
    }
    
    // Network value change callbacks (replaces Fusion's FixedUpdateNetwork)
    private void OnPositionChanged(Vector3 previousValue, Vector3 newValue)
    {
        terrainTransform.position = newValue;
    }
    
    private void OnRotationChanged(Quaternion previousValue, Quaternion newValue)
    {
        terrainTransform.rotation = newValue;
    }
    
    private void OnScaleChanged(float previousValue, float newValue)
    {
        terrainTransform.localScale = Vector3.one * newValue;
    }
    
    private void OnManipulationStateChanged(bool previousValue, bool newValue)
    {
        UpdateManipulationIndicators();
    }
    
    /// <summary>
    /// Call this when user starts grabbing the terrain
    /// </summary>
    public void StartManipulation()
    {
        if (!isBeingManipulated.Value)
        {
            isLocallyGrabbed = true;
            RequestStateAuthorityServerRpc();
        }
    }
    
    /// <summary>
    /// Call this when user releases the terrain
    /// </summary>
    public void EndManipulation()
    {
        isLocallyGrabbed = false;
        ReleaseStateAuthorityServerRpc();
    }
    
    /// <summary>
    /// Zoom the terrain (pinch gesture or controller trigger)
    /// </summary>
    public void ZoomTerrain(float zoomDelta)
    {
        if (!IsOwner) return;
        
        float newScale = Mathf.Clamp(
            networkedScale.Value + zoomDelta * zoomSpeed,
            zoomLimits.x,
            zoomLimits.y
        );
        
        UpdateScaleServerRpc(newScale);
    }
    
    /// <summary>
    /// Pan the terrain (hand/controller movement)
    /// </summary>
    public void PanTerrain(Vector3 handPosition)
    {
        if (!IsOwner) return;
        
        if (isLocallyGrabbed)
        {
            Vector3 delta = handPosition - lastHandPosition;
            Vector3 newPosition = networkedPosition.Value + delta * panSpeed;
            
            // Constrain panning to a radius
            if (newPosition.magnitude <= panRadius)
            {
                UpdatePositionServerRpc(newPosition);
            }
        }
        
        lastHandPosition = handPosition;
    }
    
    /// <summary>
    /// Rotate the terrain (twist gesture or controller rotation)
    /// </summary>
    public void RotateTerrain(float rotationDelta)
    {
        if (!IsOwner) return;
        
        Quaternion rotation = Quaternion.Euler(0, rotationDelta * rotateSpeed * Time.deltaTime, 0);
        Quaternion newRotation = networkedRotation.Value * rotation;
        UpdateRotationServerRpc(newRotation);
    }
    
    /// <summary>
    /// Two-handed rotation using hand positions
    /// </summary>
    public void RotateTerrainTwoHanded(Vector3 hand1Pos, Vector3 hand2Pos)
    {
        if (!IsOwner) return;
        
        Vector3 handVector = hand2Pos - hand1Pos;
        float angle = Mathf.Atan2(handVector.z, handVector.x) * Mathf.Rad2Deg;
        Quaternion newRotation = Quaternion.Euler(0, angle, 0);
        UpdateRotationServerRpc(newRotation);
    }
    
    /// <summary>
    /// Pinch-to-zoom using two hands
    /// </summary>
    public void PinchZoom(Vector3 hand1Pos, Vector3 hand2Pos, bool isInitial = false)
    {
        if (!IsOwner) return;
        
        float currentDistance = Vector3.Distance(hand1Pos, hand2Pos);
        
        if (isInitial)
        {
            initialPinchDistance = new Vector2(currentDistance, networkedScale.Value);
            return;
        }
        
        float scaleMultiplier = currentDistance / initialPinchDistance.x;
        float newScale = Mathf.Clamp(
            initialPinchDistance.y * scaleMultiplier,
            zoomLimits.x,
            zoomLimits.y
        );
        
        UpdateScaleServerRpc(newScale);
    }
    
    // ServerRPCs to request and release authority (adapted from Fusion RPCs)
    [ServerRpc(RequireOwnership = false)]
    private void RequestStateAuthorityServerRpc(ServerRpcParams rpcParams = default)
    {
        if (!isBeingManipulated.Value)
        {
            // Grant ownership to the requesting client (adapted from Fusion's AssignInputAuthority)
            NetworkObject.ChangeOwnership(rpcParams.Receive.SenderClientId);
            isBeingManipulated.Value = true;
            currentManipulatorId.Value = rpcParams.Receive.SenderClientId;
        }
    }
    
    [ServerRpc(RequireOwnership = false)]
    private void ReleaseStateAuthorityServerRpc(ServerRpcParams rpcParams = default)
    {
        if (currentManipulatorId.Value == rpcParams.Receive.SenderClientId)
        {
            // Remove ownership (adapted from Fusion's RemoveInputAuthority)
            NetworkObject.RemoveOwnership();
            isBeingManipulated.Value = false;
            currentManipulatorId.Value = 0;
        }
    }
    
    // ServerRPCs to update networked values
    [ServerRpc(RequireOwnership = true)]
    private void UpdatePositionServerRpc(Vector3 newPosition)
    {
        networkedPosition.Value = newPosition;
    }
    
    [ServerRpc(RequireOwnership = true)]
    private void UpdateRotationServerRpc(Quaternion newRotation)
    {
        networkedRotation.Value = newRotation;
    }
    
    [ServerRpc(RequireOwnership = true)]
    private void UpdateScaleServerRpc(float newScale)
    {
        networkedScale.Value = newScale;
    }
    
    /// <summary>
    /// Visual feedback showing who's manipulating the terrain
    /// </summary>
    private void UpdateManipulationIndicators()
    {
        if (manipulationIndicator != null)
        {
            manipulationIndicator.SetActive(isBeingManipulated.Value);
            
            if (isBeingManipulated.Value && currentManipulatorId.Value != 0)
            {
                // Set indicator color based on user (adapted from Fusion's PlayerRef.PlayerId)
                int colorIndex = (int)(currentManipulatorId.Value % (ulong)userColors.Length);
                var renderer = manipulationIndicator.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.color = userColors[colorIndex];
                }
            }
        }
    }
    
    // Optional: Reset terrain to default state (adapted from Fusion RPC)
    [ServerRpc(RequireOwnership = false)]
    public void ResetTerrainServerRpc()
    {
        networkedPosition.Value = Vector3.zero;
        networkedRotation.Value = Quaternion.identity;
        networkedScale.Value = 1f;
    }
    
    // Public getters for current state
    public Vector3 NetworkedPosition => networkedPosition.Value;
    public Quaternion NetworkedRotation => networkedRotation.Value;
    public float NetworkedScale => networkedScale.Value;
    public bool IsBeingManipulated => isBeingManipulated.Value;
    public ulong CurrentManipulator => currentManipulatorId.Value;
}