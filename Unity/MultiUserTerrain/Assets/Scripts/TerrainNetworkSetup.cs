using UnityEngine;
using Unity.Netcode;

/// <summary>
/// Helper script to automatically connect TerrainInputHandler to TerrainInteractionManager
/// Also provides utility functions for terrain setup
/// </summary>
public class TerrainNetworkSetup : NetworkBehaviour
{
    [Header("Auto-find References")]
    [SerializeField] private bool autoFindTerrain = true;
    [SerializeField] private bool autoFindInputHandler = true;

    [Header("Manual References (optional)")]
    [SerializeField] private TerrainInteractionManager terrainManager;
    [SerializeField] private TerrainInputHandler inputHandler;

    public override void OnNetworkSpawn()
    {
        base.OnNetworkSpawn();

        // Only set up on local client
        if (IsOwner)
        {
            SetupReferences();
        }
    }

    private void Start()
    {
        // For non-networked testing, set up references immediately
        if (!NetworkManager.Singleton || !NetworkManager.Singleton.IsListening)
        {
            SetupReferences();
        }
    }

    private void SetupReferences()
    {
        // Find TerrainInteractionManager if needed
        if (terrainManager == null && autoFindTerrain)
        {
            terrainManager = FindObjectOfType<TerrainInteractionManager>();
            if (terrainManager != null)
            {
                Debug.Log("TerrainInteractionManager found automatically");
            }
            else
            {
                Debug.LogWarning("TerrainInteractionManager not found in scene!");
                return;
            }
        }

        // Find TerrainInputHandler if needed
        if (inputHandler == null && autoFindInputHandler)
        {
            // Try to find on this GameObject first
            inputHandler = GetComponent<TerrainInputHandler>();

            // If not found, search in scene
            if (inputHandler == null)
            {
                inputHandler = FindObjectOfType<TerrainInputHandler>();
            }

            if (inputHandler != null)
            {
                Debug.Log("TerrainInputHandler found automatically");
            }
            else
            {
                Debug.LogWarning("TerrainInputHandler not found!");
                return;
            }
        }

        // Connect input handler to terrain manager using reflection
        if (terrainManager != null && inputHandler != null)
        {
            // Use reflection to set the private terrainManager field in TerrainInputHandler
            var field = typeof(TerrainInputHandler).GetField("terrainManager",
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.Instance);

            if (field != null)
            {
                field.SetValue(inputHandler, terrainManager);
                Debug.Log("Successfully connected TerrainInputHandler to TerrainInteractionManager");
            }
            else
            {
                Debug.LogError("Could not find terrainManager field in TerrainInputHandler. " +
                              "Make sure the field is serialized or provide a public setter method.");
            }
        }
    }

    /// <summary>
    /// Manually connect terrain manager to input handler
    /// Call this if auto-find fails or you want manual control
    /// </summary>
    public void ManuallyConnectComponents(TerrainInteractionManager terrain, TerrainInputHandler input)
    {
        terrainManager = terrain;
        inputHandler = input;
        SetupReferences();
    }

    /// <summary>
    /// Get the terrain manager reference
    /// </summary>
    public TerrainInteractionManager GetTerrainManager()
    {
        return terrainManager;
    }

    /// <summary>
    /// Get the input handler reference
    /// </summary>
    public TerrainInputHandler GetInputHandler()
    {
        return inputHandler;
    }
}