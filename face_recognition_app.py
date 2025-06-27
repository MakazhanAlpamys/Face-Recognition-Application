import cv2
import mediapipe as mp
import numpy as np
import time

class FaceRecognitionApp:
    def __init__(self):
        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face detection model
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range detection
            min_detection_confidence=0.5)
        
        # Initialize face mesh model for landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Face database - will store face embeddings of recognized users
        self.face_database = {}
        self.current_user_id = None
        self.detection_cooldown = 0
        
    def calculate_face_encoding(self, landmarks, img_shape):
        """
        Create a simple face encoding from landmarks
        """
        # Convert landmarks to normalized coordinates
        h, w = img_shape[:2]
        points = []
        
        # Use a subset of key facial landmarks for the encoding
        key_points = [1, 33, 61, 199, 263, 291, 362, 397]
        
        for idx in key_points:
            if idx < len(landmarks.landmark):
                point = landmarks.landmark[idx]
                points.append([point.x, point.y, point.z])
        
        return np.array(points).flatten()
    
    def compare_faces(self, known_encoding, encoding_to_check, tolerance=0.4):
        """
        Compare face encodings to determine if they are the same person
        """
        # Calculate Euclidean distance
        dist = np.linalg.norm(known_encoding - encoding_to_check)
        return dist < tolerance, dist
    
    def register_new_user(self, encoding):
        """
        Register a new user in the face database
        """
        user_id = f"user_{len(self.face_database) + 1}"
        self.face_database[user_id] = encoding
        print(f"New user registered: {user_id}")
        return user_id
    
    def identify_user(self, face_encoding):
        """
        Check if the face matches any registered user
        """
        if not self.face_database:
            return self.register_new_user(face_encoding), True
        
        best_match = None
        min_distance = float('inf')
        
        for user_id, stored_encoding in self.face_database.items():
            is_match, distance = self.compare_faces(stored_encoding, face_encoding)
            if is_match and distance < min_distance:
                min_distance = distance
                best_match = user_id
        
        if best_match:
            return best_match, False
        else:
            return self.register_new_user(face_encoding), True
    
    def run(self):
        # Start video capture from webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Error: Failed to read frame from webcam")
                continue
            
            # Flip the image horizontally for selfie view
            image = cv2.flip(image, 1)
            
            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process for face detection
            face_detection_results = self.face_detection.process(image_rgb)
            
            # Draw face detection results
            if face_detection_results.detections:
                for detection in face_detection_results.detections:
                    self.mp_drawing.draw_detection(image, detection)
                    
                    # Get bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Process for face mesh
            face_mesh_results = self.face_mesh.process(image_rgb)
            
            # Draw face mesh landmarks
            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    # Calculate face encoding
                    if self.detection_cooldown <= 0:
                        face_encoding = self.calculate_face_encoding(face_landmarks, image.shape)
                        user_id, is_new = self.identify_user(face_encoding)
                        self.current_user_id = user_id
                        self.detection_cooldown = 30  # Cooldown for 30 frames
                    else:
                        self.detection_cooldown -= 1
            
            # Display user ID on screen
            if self.current_user_id:
                cv2.putText(image, f"User: {self.current_user_id}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Display face distance information
                if face_mesh_results.multi_face_landmarks and self.current_user_id in self.face_database:
                    face_landmarks = face_mesh_results.multi_face_landmarks[0]
                    face_encoding = self.calculate_face_encoding(face_landmarks, image.shape)
                    _, distance = self.compare_faces(self.face_database[self.current_user_id], face_encoding)
                    cv2.putText(image, f"Match confidence: {1.0 - distance:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Draw instructions
            cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Press 'r' to reset face database", (10, image.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display the image
            cv2.imshow('Face Recognition App', image)
            
            # Handle key presses
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.face_database = {}
                self.current_user_id = None
                print("Face database reset")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run() 