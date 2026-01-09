import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import json
import threading
import time

print("🚀 Starting Plant Disease Detector - FINAL VERSION")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import dependencies
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    messagebox.showerror("Error", "Pillow not installed")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

class PlantDiseaseDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detector - Professional")
        self.root.geometry("1200x900")  # Larger window
        self.root.configure(bg='#f5f5f5')
        
        # Make window resizable
        self.root.minsize(1000, 700)
        
        # Model and state variables
        self.model = None
        self.class_mapping = {}
        self.camera = None
        self.is_camera_active = False
        self.current_image = None
        self.current_image_path = None
        
        # Setup GUI
        self.setup_gui()
        
        # Load model
        self.root.after(500, self.load_model)
    
    def setup_gui(self):
        """Setup the graphical user interface with larger results area"""
        # Configure styles
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground='#2e7d32')
        style.configure('Large.TButton', font=('Arial', 11), padding=(15, 8))
        style.configure('Results.TFrame', background='white')
        
        # Main container using PanedWindow for resizable panels
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls and Image
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Results
        right_frame = ttk.Frame(main_paned, style='Results.TFrame')
        main_paned.add(right_frame, weight=1)
        
        # Configure paned window to give more space to results initially
        main_paned.sashpos(0, 600)
        
        # ===== LEFT PANEL =====
        # Header
        header_frame = ttk.Frame(left_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="🌱 Plant Disease Detector", style='Title.TLabel')
        title_label.pack()
        
        # Controls frame
        controls_frame = ttk.LabelFrame(left_frame, text="Image Controls", padding="15")
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Buttons grid
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X)
        
        self.upload_btn = ttk.Button(button_frame, text="📁 Upload Image", 
                                    command=self.upload_image, width=18, style='Large.TButton')
        self.upload_btn.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        
        self.camera_btn = ttk.Button(button_frame, text="📷 Open Camera", 
                                    command=self.toggle_camera, width=18, style='Large.TButton',
                                    state='normal' if CV2_AVAILABLE else 'disabled')
        self.camera_btn.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        self.analyze_btn = ttk.Button(button_frame, text="🔍 Analyze Image", 
                                     command=self.analyze_image, width=18, style='Large.TButton',
                                     state='disabled')
        self.analyze_btn.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        
        self.clear_btn = ttk.Button(button_frame, text="🗑️ Clear All", 
                                   command=self.clear_all, width=18, style='Large.TButton')
        self.clear_btn.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Configure grid weights for responsive buttons
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        # Image display area
        image_frame = ttk.LabelFrame(left_frame, text="Image Preview", padding="15")
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image display with scrollbars
        image_container = ttk.Frame(image_frame)
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbars for large images
        self.canvas = tk.Canvas(image_container, bg='white', highlightthickness=1, highlightbackground='#cccccc')
        v_scrollbar = ttk.Scrollbar(image_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(image_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout for canvas and scrollbars
        self.canvas.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        image_container.rowconfigure(0, weight=1)
        image_container.columnconfigure(0, weight=1)
        
        # Image label inside canvas
        self.image_label = ttk.Label(self.canvas, 
                                    text="👆 Upload an image or use camera\n\n"
                                         "Supported formats: JPG, PNG, BMP, TIFF\n\n"
                                         "For best results, use clear images of plant leaves",
                                    background='white',
                                    justify=tk.CENTER,
                                    font=('Arial', 12),
                                    anchor='center')
        
        self.canvas_window = self.canvas.create_window(0, 0, anchor='nw', window=self.image_label)
        
        # Update scroll region when image changes
        def configure_canvas(event):
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        
        self.image_label.bind('<Configure>', configure_canvas)
        
        # ===== RIGHT PANEL - RESULTS =====
        # Results header
        results_header = ttk.Frame(right_frame)
        results_header.pack(fill=tk.X, pady=(0, 10))
        
        results_title = ttk.Label(results_header, text="🔍 Analysis Results", 
                                 font=('Arial', 16, 'bold'), foreground='#1976d2')
        results_title.pack()
        
        # Results container with larger area
        results_container = ttk.Frame(right_frame)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with larger font and more space
        self.results_text = tk.Text(results_container, 
                                   wrap=tk.WORD, 
                                   font=('Arial', 12),
                                   padx=20, 
                                   pady=20,
                                   bg='#fafafa',
                                   relief='flat')
        
        # Configure tags for better formatting
        self.results_text.tag_configure('title', font=('Arial', 14, 'bold'), foreground='#2e7d32')
        self.results_text.tag_configure('header', font=('Arial', 12, 'bold'), foreground='#1976d2')
        self.results_text.tag_configure('disease', font=('Arial', 11, 'bold'), foreground='#d32f2f')
        self.results_text.tag_configure('confidence_high', foreground='#388e3c')
        self.results_text.tag_configure('confidence_medium', foreground='#f57c00')
        self.results_text.tag_configure('confidence_low', foreground='#d32f2f')
        self.results_text.tag_configure('normal', font=('Arial', 11))
        
        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(results_container, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        # Pack results area
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar at bottom
        status_frame = ttk.Frame(self.root, relief='sunken', borderwidth=1)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar()
        self.status_var.set("🟢 Ready - Upload an image to start analysis")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, 
                              padding=(10, 5), font=('Arial', 9))
        status_bar.pack(fill=tk.X)
    
    def load_model(self):
        """Load the YOLO model"""
        self.status_var.set("🔄 Loading plant disease detection model...")
        
        if not os.path.exists("best_yolo_plant_disease.pt"):
            self.status_var.set("❌ Model file not found")
            return
            
        if not os.path.exists("class_mapping.json"):
            self.status_var.set("❌ Class mapping file not found")
            return
            
        if not YOLO_AVAILABLE:
            self.status_var.set("❌ YOLO not available")
            return
            
        try:
            with open("class_mapping.json", 'r', encoding='utf-8') as f:
                class_mapping_raw = json.load(f)
                self.class_mapping = {}
                for k, v in class_mapping_raw.items():
                    try:
                        self.class_mapping[int(k)] = v
                    except ValueError:
                        self.class_mapping[k] = v
            
            self.model = YOLO("best_yolo_plant_disease.pt")
            self.model.to('cpu')
                
            self.status_var.set(f"✅ Model loaded - {len(self.class_mapping)} plant diseases")
            
        except Exception as e:
            self.status_var.set(f"❌ Failed to load model: {str(e)}")
    
    def upload_image(self):
        """Upload an image file"""
        if not PIL_AVAILABLE:
            messagebox.showerror("Error", "PIL/Pillow not available")
            return
            
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Plant Image",
            filetypes=file_types
        )
        
        if file_path:
            self.process_uploaded_image(file_path)
    
    def process_uploaded_image(self, file_path):
        """Process and display uploaded image"""
        self.status_var.set("🔄 Loading image...")
        
        try:
            image = Image.open(file_path)
            self.current_image = image
            self.current_image_path = file_path
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Display image
            self.display_image(image)
            
            self.analyze_btn.config(state='normal')
            self.status_var.set(f"✅ Image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {str(e)}")
            self.status_var.set("❌ Failed to load image")
    
    def display_image(self, image):
        """Display PIL image in the canvas"""
        # Calculate display size (larger for new layout)
        display_width = 500
        display_height = 400
        
        img_width, img_height = image.size
        img_ratio = img_width / img_height
        display_ratio = display_width / display_height
        
        if img_ratio > display_ratio:
            new_width = display_width
            new_height = int(display_width / img_ratio)
        else:
            new_height = display_height
            new_width = int(display_height * img_ratio)
        
        # Resize image
        image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image_resized)
        
        # Update display
        self.image_label.configure(image=photo, text='')
        self.image_label.image = photo
        
        # Update canvas scroll region
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not CV2_AVAILABLE:
            messagebox.showerror("Error", "OpenCV not available for camera")
            return
            
        if not self.is_camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                return
                
            self.is_camera_active = True
            self.camera_btn.config(text="📷 Close Camera")
            self.status_var.set("📷 Camera active - Position plant and click Analyze")
            
            self.camera_thread = threading.Thread(target=self.camera_preview, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_camera_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.camera_btn.config(text="📷 Open Camera")
        self.status_var.set("Camera stopped")
        
        if hasattr(self, 'camera_photo'):
            self.image_label.configure(image='')
            self.image_label.configure(text="👆 Upload an image or use camera\n\n"
                                          "Supported formats: JPG, PNG, BMP, TIFF\n\n"
                                          "For best results, use clear images of plant leaves")
    
    def camera_preview(self):
        """Camera preview thread"""
        while self.is_camera_active and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret and PIL_AVAILABLE:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    self.root.after(0, self.update_camera_display, image, frame)
                time.sleep(0.03)
            except Exception:
                break
    
    def update_camera_display(self, image, frame):
        """Update camera display in main thread"""
        if self.is_camera_active:
            self.display_image(image)
            self.current_camera_frame = frame
            self.analyze_btn.config(state='normal')
    
    def analyze_image(self):
        """Analyze the current image"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
            
        self.status_var.set("🔍 Analyzing plant leaf for diseases...")
        self.analyze_btn.config(state='disabled')
        
        analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        analysis_thread.start()
    
    def run_analysis(self):
        """Run analysis in background thread"""
        try:
            if hasattr(self, 'current_camera_frame') and CV2_AVAILABLE:
                image_source = self.current_camera_frame
            elif self.current_image_path:
                image_source = self.current_image_path
            else:
                self.root.after(0, lambda: self.status_var.set("❌ No image available"))
                return
            
            results = self.model.predict(
                source=image_source,
                conf=0.5,
                imgsz=640,
                device='cpu',
                verbose=False,
                save=False
            )
            
            predictions = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        class_name = self.class_mapping.get(class_id, 
                                                          self.class_mapping.get(str(class_id), 
                                                                                f"Disease_{class_id}"))
                        
                        predictions.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'confidence_percent': f"{confidence:.1%}"
                        })
            
            self.root.after(0, self.show_results, predictions)
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))
    
    def show_results(self, predictions):
        """Show analysis results with beautiful formatting"""
        self.results_text.delete(1.0, tk.END)
        
        # Insert title
        self.results_text.insert(tk.END, "PLANT DISEASE ANALYSIS REPORT\n", 'title')
        self.results_text.insert(tk.END, "=" * 50 + "\n\n", 'normal')
        
        if predictions:
            # Summary
            self.results_text.insert(tk.END, "SUMMARY\n", 'header')
            self.results_text.insert(tk.END, f"• Total detections: {len(predictions)}\n", 'normal')
            self.results_text.insert(tk.END, f"• Analysis time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n", 'normal')
            
            # Detections
            self.results_text.insert(tk.END, "DETECTED DISEASES\n", 'header')
            self.results_text.insert(tk.END, "─" * 30 + "\n\n", 'normal')
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            for i, pred in enumerate(predictions, 1):
                # Determine confidence level and color
                if pred['confidence'] > 0.8:
                    confidence_tag = 'confidence_high'
                    level = "HIGH"
                    emoji = "🟢"
                elif pred['confidence'] > 0.5:
                    confidence_tag = 'confidence_medium'
                    level = "MEDIUM" 
                    emoji = "🟡"
                else:
                    confidence_tag = 'confidence_low'
                    level = "LOW"
                    emoji = "🔴"
                
                # Insert disease information
                self.results_text.insert(tk.END, f"{i}. ", 'normal')
                self.results_text.insert(tk.END, f"{pred['class_name']}\n", 'disease')
                self.results_text.insert(tk.END, f"   Confidence: {pred['confidence_percent']} ", 'normal')
                self.results_text.insert(tk.END, f"({level}) {emoji}\n\n", confidence_tag)
            
            # Recommendations
            self.results_text.insert(tk.END, "RECOMMENDATIONS\n", 'header')
            self.results_text.insert(tk.END, "─" * 30 + "\n\n", 'normal')
            self.results_text.insert(tk.END, "• Consult with agricultural expert for proper treatment\n", 'normal')
            self.results_text.insert(tk.END, "• Isolate affected plants if possible\n", 'normal')
            self.results_text.insert(tk.END, "• Follow recommended treatment protocols\n", 'normal')
            self.results_text.insert(tk.END, "• Monitor plant health regularly\n", 'normal')
            self.results_text.insert(tk.END, "• Maintain proper plant care practices\n\n", 'normal')
            
        else:
            # Healthy plant results
            self.results_text.insert(tk.END, "SUMMARY\n", 'header')
            self.results_text.insert(tk.END, "• Status: HEALTHY 🌱\n", 'normal')
            self.results_text.insert(tk.END, "• No diseases detected\n", 'normal')
            self.results_text.insert(tk.END, f"• Analysis time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n", 'normal')
            
            self.results_text.insert(tk.END, "RECOMMENDATIONS\n", 'header')
            self.results_text.insert(tk.END, "─" * 30 + "\n\n", 'normal')
            self.results_text.insert(tk.END, "✅ Continue current plant care practices\n", 'normal')
            self.results_text.insert(tk.END, "✅ Maintain proper watering schedule\n", 'normal')
            self.results_text.insert(tk.END, "✅ Ensure adequate sunlight exposure\n", 'normal')
            self.results_text.insert(tk.END, "✅ Monitor for any changes in plant health\n", 'normal')
            self.results_text.insert(tk.END, "✅ Regular inspection is recommended\n\n", 'normal')
        
        # Footer
        self.results_text.insert(tk.END, "NOTE\n", 'header')
        self.results_text.insert(tk.END, "This analysis is provided by AI and should be verified by agricultural experts for critical decisions.\n", 'normal')
        
        # Scroll to top
        self.results_text.see(1.0)
        
        self.status_var.set("✅ Analysis complete")
        self.analyze_btn.config(state='normal')
    
    def show_error(self, error_msg):
        """Show error message"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "ANALYSIS ERROR\n", 'title')
        self.results_text.insert(tk.END, "=" * 50 + "\n\n", 'normal')
        self.results_text.insert(tk.END, f"Error: {error_msg}\n\n", 'normal')
        self.results_text.insert(tk.END, "Please check:\n", 'header')
        self.results_text.insert(tk.END, "• Image format and quality\n", 'normal')
        self.results_text.insert(tk.END, "• Model file integrity\n", 'normal')
        self.results_text.insert(tk.END, "• System resources\n", 'normal')
        
        self.status_var.set("❌ Analysis failed")
        self.analyze_btn.config(state='normal')
    
    def clear_all(self):
        """Clear everything"""
        self.current_image = None
        self.current_image_path = None
        
        if hasattr(self, 'current_camera_frame'):
            del self.current_camera_frame
        
        self.image_label.configure(image='')
        self.image_label.configure(
            text="👆 Upload an image or use camera\n\n"
                 "Supported formats: JPG, PNG, BMP, TIFF\n\n"
                 "For best results, use clear images of plant leaves",
            background='white'
        )
        
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("🟢 Cleared - Ready for new analysis")
        self.analyze_btn.config(state='disabled')
        
        if self.is_camera_active:
            self.stop_camera()

def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        app = PlantDiseaseDetector(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{str(e)}")

if __name__ == "__main__":
    main()