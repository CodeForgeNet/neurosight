import cv2
import time
from neurosight.detector import MotionDetector, NeuroSightConfig

def main():
    # Use the looming preset for collision warning detection
    config = NeuroSightConfig.preset_looming(tile_size=8)
    detector = MotionDetector(config=config)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print("Starting NeuroSight Steering Demo. Press 'q' to quit.")
    
    for _ in range(5):
        cap.read()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize for performance on edge devices
        frame = cv2.resize(frame, (640, 480))
        
        # Process the frame continuously
        result = detector.update(frame, timestamp=time.time())
        
        if result is not None:
            # Print looming/TTC
            if result.looming_score is not None and result.looming_score > 0.05:
                print(f"[WARNING] Collision imminent! TTC: {result.time_to_contact:.2f} frames")
                
            # Print Steering
            steering = result.steering_signal
            if steering is not None and steering['forward_caution'] > 0.3:
                direction = "RIGHT" if steering['turn'] > 0 else "LEFT"
                strength = abs(steering['turn'])
                if strength > 0.3:
                    print(f"-> Steer {direction} (Strength: {strength:.2f})")
            
            # Show original
            cv2.imshow('NeuroSight Robotics', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
