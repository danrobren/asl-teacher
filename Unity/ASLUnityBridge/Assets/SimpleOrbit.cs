using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public class SimpleOrbit : MonoBehaviour
{
    public Transform target;
    public float distance = 2.0f;
    public float rotateSpeed = 180f;
    public float zoomSpeed = 2f;
    float yaw, pitch = 20f;

    void LateUpdate()
    {
        if (!target) return;

        float dx = 0f, dy = 0f, scroll = 0f;
        bool dragging = false;

#if ENABLE_INPUT_SYSTEM
        // New Input System
        if (Mouse.current != null)
        {
            dragging = Mouse.current.leftButton.isPressed;
            Vector2 delta = Mouse.current.delta.ReadValue();
            dx = delta.x * Time.deltaTime;
            dy = delta.y * Time.deltaTime;
            scroll = Mouse.current.scroll.ReadValue().y / 120f; // roughly steps
        }
#else
        // Old Input Manager
        dragging = Input.GetMouseButton(0);
        dx = Input.GetAxis("Mouse X");
        dy = Input.GetAxis("Mouse Y");
        scroll = Input.GetAxis("Mouse ScrollWheel");
#endif

        if (dragging)
        {
            yaw   += dx * rotateSpeed * 0.02f;
            pitch -= dy * rotateSpeed * 0.02f;
            pitch = Mathf.Clamp(pitch, -10f, 80f);
        }

        distance = Mathf.Clamp(distance - scroll * zoomSpeed, 0.5f, 6f);

        var rot = Quaternion.Euler(pitch, yaw, 0);
        var pos = target.position + rot * (Vector3.back * distance);
        transform.SetPositionAndRotation(pos, rot);
    }
}
