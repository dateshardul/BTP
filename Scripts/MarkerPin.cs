using UnityEngine;
using Unity.Netcode;
using TMPro;

/// <summary>
/// Individual marker pin (like Google Maps pin)
/// Red rounded top, vertical line, networked label
/// </summary>
public class MarkerPin : NetworkBehaviour
{
    [Header("Visual Components")]
    [SerializeField] private GameObject pinHead;  // Red rounded sphere
    [SerializeField] private GameObject pinStick;  // Vertical cylinder
    [SerializeField] private TextMeshPro labelText;  // 3D text label

    [Header("Pin Settings")]
    [SerializeField] private Color pinColor = Color.red;
    [SerializeField] private float pinHeight = 0.3f;  // Total height in meters
    [SerializeField] private float headRadius = 0.05f;  // Sphere radius
    [SerializeField] private float stickRadius = 0.005f;  // Cylinder radius
    [SerializeField] private bool billboardLabel = true;  // Label always faces camera

    [Header("Label Settings")]
    [SerializeField] private float labelFontSize = 0.1f;
    [SerializeField] private Vector3 labelOffset = new Vector3(0, 0.35f, 0);  // Offset above pin

    // Networked properties
    private NetworkVariable<FixedString32Bytes> markerLabel = new NetworkVariable<FixedString32Bytes>(
        new FixedString32Bytes(""),
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    private NetworkVariable<ulong> creatorClientId = new NetworkVariable<ulong>(
        0,
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Server
    );

    private Camera mainCamera;

    private void Start()
    {
        // Create pin geometry if not assigned
        if (pinHead == null || pinStick == null)
        {
            CreatePinGeometry();
        }

        // Setup label
        if (labelText == null)
        {
            CreateLabel();
        }

        // Find main camera for billboarding
        mainCamera = Camera.main;
    }

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Subscribe to label changes
        markerLabel.OnValueChanged += OnLabelChanged;

        // Initial label update
        UpdateLabel();
    }

    public override void OnNetworkDespawn()
    {
        if (markerLabel != null)
        {
            markerLabel.OnValueChanged -= OnLabelChanged;
        }
        base.OnNetworkDespawn();
    }

    private void Update()
    {
        // Billboard label to face camera
        if (billboardLabel && labelText != null && mainCamera != null)
        {
            labelText.transform.LookAt(mainCamera.transform);
            labelText.transform.Rotate(0, 180, 0);  // Flip to face correctly
        }
    }

    /// <summary>
    /// Create pin geometry procedurally
    /// </summary>
    private void CreatePinGeometry()
    {
        // Create pin head (red sphere at top)
        if (pinHead == null)
        {
            pinHead = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            pinHead.name = "PinHead";
            pinHead.transform.SetParent(transform);
            pinHead.transform.localPosition = new Vector3(0, pinHeight - headRadius, 0);
            pinHead.transform.localScale = Vector3.one * (headRadius * 2);

            // Set color
            Renderer headRenderer = pinHead.GetComponent<Renderer>();
            if (headRenderer != null)
            {
                headRenderer.material.color = pinColor;
            }

            // Remove collider (we don't need physics)
            Destroy(pinHead.GetComponent<Collider>());
        }

        // Create pin stick (vertical cylinder)
        if (pinStick == null)
        {
            pinStick = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            pinStick.name = "PinStick";
            pinStick.transform.SetParent(transform);

            // Position stick between ground and head
            float stickHeight = pinHeight - headRadius;
            pinStick.transform.localPosition = new Vector3(0, stickHeight / 2, 0);
            pinStick.transform.localScale = new Vector3(stickRadius * 2, stickHeight / 2, stickRadius * 2);

            // Set color (slightly darker than head)
            Renderer stickRenderer = pinStick.GetComponent<Renderer>();
            if (stickRenderer != null)
            {
                stickRenderer.material.color = pinColor * 0.8f;
            }

            // Remove collider
            Destroy(pinStick.GetComponent<Collider>());
        }

        // Add shadow plane (optional - small circle at base)
        GameObject shadow = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        shadow.name = "PinShadow";
        shadow.transform.SetParent(transform);
        shadow.transform.localPosition = new Vector3(0, 0.001f, 0);  // Just above ground
        shadow.transform.localScale = new Vector3(headRadius * 1.5f, 0.001f, headRadius * 1.5f);

        Renderer shadowRenderer = shadow.GetComponent<Renderer>();
        if (shadowRenderer != null)
        {
            shadowRenderer.material.color = new Color(0, 0, 0, 0.3f);  // Semi-transparent black
        }
        Destroy(shadow.GetComponent<Collider>());
    }

    /// <summary>
    /// Create 3D text label
    /// </summary>
    private void CreateLabel()
    {
        GameObject labelObj = new GameObject("Label");
        labelObj.transform.SetParent(transform);
        labelObj.transform.localPosition = labelOffset;

        labelText = labelObj.AddComponent<TextMeshPro>();
        labelText.fontSize = labelFontSize;
        labelText.alignment = TextAlignmentOptions.Center;
        labelText.color = Color.white;
        labelText.outlineColor = Color.black;
        labelText.outlineWidth = 0.2f;

        // Add background panel for better readability
        GameObject background = GameObject.CreatePrimitive(PrimitiveType.Quad);
        background.name = "LabelBackground";
        background.transform.SetParent(labelObj.transform);
        background.transform.localPosition = new Vector3(0, 0, 0.01f);  // Slightly behind text
        background.transform.localScale = new Vector3(0.2f, 0.05f, 1);

        Renderer bgRenderer = background.GetComponent<Renderer>();
        if (bgRenderer != null)
        {
            bgRenderer.material.color = new Color(0, 0, 0, 0.7f);  // Semi-transparent black
        }
        Destroy(background.GetComponent<Collider>());
    }

    /// <summary>
    /// Set marker label (server only)
    /// </summary>
    [ClientRpc]
    public void SetLabelClientRpc(string label)
    {
        if (IsServer)
        {
            markerLabel.Value = new FixedString32Bytes(label);
        }
    }

    /// <summary>
    /// Set creator client ID (server only)
    /// </summary>
    [ClientRpc]
    public void SetCreatorClientRpc(ulong clientId)
    {
        if (IsServer)
        {
            creatorClientId.Value = clientId;
        }
    }

    private void OnLabelChanged(FixedString32Bytes previousValue, FixedString32Bytes newValue)
    {
        UpdateLabel();
    }

    private void UpdateLabel()
    {
        if (labelText != null)
        {
            labelText.text = markerLabel.Value.ToString();
        }
    }

    /// <summary>
    /// Highlight pin (for selection or hover)
    /// </summary>
    public void SetHighlighted(bool highlighted)
    {
        if (pinHead != null)
        {
            Renderer renderer = pinHead.GetComponent<Renderer>();
            if (renderer != null)
            {
                if (highlighted)
                {
                    renderer.material.color = Color.yellow;
                    transform.localScale = Vector3.one * 1.2f;  // Slightly bigger
                }
                else
                {
                    renderer.material.color = pinColor;
                    transform.localScale = Vector3.one;
                }
            }
        }
    }

    /// <summary>
    /// Pulse animation (for newly placed markers)
    /// </summary>
    public void PlayPlacementAnimation()
    {
        // Simple scale animation
        LeanTween.scale(gameObject, Vector3.one * 1.3f, 0.2f)
            .setEaseOutBack()
            .setOnComplete(() =>
            {
                LeanTween.scale(gameObject, Vector3.one, 0.2f).setEaseInBack();
            });
    }

    // Public properties
    public string Label => markerLabel.Value.ToString();
    public ulong CreatorId => creatorClientId.Value;
    public Vector3 Position => transform.position;

    // Cleanup
    private void OnDestroy()
    {
        // Cancel any running animations
        LeanTween.cancel(gameObject);
    }
}