import express, { Request, Response } from 'express';
import cors from 'cors';
import multer from 'multer';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs'; // Import the file system module

const app = express();
const PORT = 5001;

// --- Middleware Setup ---
app.use(cors());
app.use(express.json());

// --- Multer Configuration for File Uploads ---
// Ensure the 'uploads' directory exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsDir),
  filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname)),
});

const upload = multer({ storage });


// --- API Endpoint for Prediction ---
app.post('/predict', upload.single('image'), (req: Request, res: Response) => {
  // --- Start of Debugging ---
  console.log('--- PREDICT ROUTE HIT ---');

  if (!req.file) {
    console.error('Error: No file object received from multer.');
    return res.status(400).json({ error: 'No image file provided' });
  }

  // Log all the information multer gives us about the file
  console.log('Multer file object:', req.file);

  // This is the absolute path where multer saved the file
  const absolutePath = req.file.path;
  console.log('Absolute path of saved file:', absolutePath);
  
  // Verify if the file actually exists at that path
  if (fs.existsSync(absolutePath)) {
      console.log('File verification successful: File exists at path.');
  } else {
      console.error('File verification FAILED: File does not exist at path.');
      return res.status(500).json({ error: 'Server failed to save the uploaded file.' });
  }
  // --- End of Debugging ---

  // Get the absolute path to the Python script
  const pythonScriptPath = path.resolve(__dirname, '../../../prediction_service/predict.py');
  
  // Spawn the child process to run the Python script
  const pythonProcess = spawn('python', [pythonScriptPath, absolutePath]);

  let pythonOutput = '';
  pythonProcess.stdout.on('data', (data) => {
    pythonOutput += data.toString();
  });

  let pythonError = '';
  pythonProcess.stderr.on('data', (data) => {
    pythonError += data.toString();
    console.error(`Python STDERR: ${data}`); // Log errors from Python in real-time
  });

  // Handle what happens when the Python script finishes
  pythonProcess.on('close', (code) => {
    console.log(`Python process finished with exit code ${code}`);
    if (code === 0) { // Code 0 means success
      try {
        const result = JSON.parse(pythonOutput);
        console.log('Received from Python:', result);
        res.json(result);
      } catch (e) {
        console.error('Failed to parse JSON from Python:', pythonOutput);
        res.status(500).json({ error: 'Failed to parse prediction result.' });
      }
    } else { // Any other code means an error occurred
      console.error('Python script exited with an error.');
      res.status(500).json({ error: 'An error occurred during prediction.', details: pythonError });
    }
  });
});


// --- Start the Server ---
app.listen(PORT, () => {
  console.log(`Backend server is running on http://localhost:${PORT}`);
});