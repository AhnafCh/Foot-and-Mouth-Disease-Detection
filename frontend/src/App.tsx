import React, { useState, ChangeEvent } from 'react';
import './App.css'; // Make sure you have the corresponding CSS file

// TypeScript interface to define the expected structure for a successful prediction
interface PredictionResult {
  prediction: string;
  confidence: string;
}

// TypeScript interface to define the structure for an error response
interface ErrorResult {
  error: string;
  details?: string; // Optional field for more detailed errors
}

// The main App component
function App() {
  // React 'state' variables to manage the component's data and behavior
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null); // URL for the image preview
  const [result, setResult] = useState<string>('Please select an image to begin.');
  const [isLoading, setIsLoading] = useState<boolean>(false); // To disable the button during classification

  /**
   * This function is called whenever the user selects a file using the input element.
   */
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    // Access the selected file from the event
    const file = event.target.files?.[0];

    if (file) {
      setSelectedFile(file);
      // Create a temporary local URL for the selected file to show a preview
      setPreview(URL.createObjectURL(file));
      setResult('Image selected. Click "Classify" to get a prediction.');
    }
  };

  /**
   * This function is called when the "Classify" button is clicked.
   * It handles sending the image to the backend and displaying the result.
   */
  const handleClassifyClick = async () => {
    if (!selectedFile) {
      alert('Please select an image file first.');
      return;
    }

    // Update the UI to show that processing has started
    setIsLoading(true);
    setResult('Uploading and classifying, please wait...');

    // FormData is the standard way to send files over HTTP
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      // Send the request to our Vercel Python serverless function.
      // The '/api/python/predict' URL is determined by our Vercel routing configuration.
      const response = await fetch('/api/python/predict', {
        method: 'POST',
        body: formData,
      });

      // Parse the JSON response from the server
      const data: PredictionResult | ErrorResult = await response.json();

      // Check if the response was an error or a successful prediction
      if ('error' in data) {
        console.error('Error from server:', data);
        setResult(`Error: ${data.error}`);
      } else {
        setResult(`Prediction: ${data.prediction} (Confidence: ${data.confidence})`);
      }
    } catch (error) {
      // This catches network errors or other issues with the fetch request itself
      console.error('Fetch Error:', error);
      setResult('A network error occurred. Please check your connection and the server status.');
    } finally {
      // This block runs regardless of success or failure, re-enabling the button
      setIsLoading(false);
    }
  };

  // This is the JSX that defines the structure and appearance of the web page
  return (
    <div className="App">
      <header className="App-header">
        <h1>FMD Detector</h1>
        <p>Upload an image of cattle to classify for Foot and Mouth Disease.</p>
        
        <div className="controls">
            <input type="file" onChange={handleFileChange} accept="image/*" />
            <button 
              onClick={handleClassifyClick} 
              disabled={!selectedFile || isLoading} // Button is disabled if no file is selected or if loading
            >
              {isLoading ? 'Processing...' : 'Classify'}
            </button>
        </div>

        <div className="result-div">{result}</div>
        
        {/* Only show the image preview element if a preview URL exists */}
        {preview && <img src={preview} alt="Selected Preview" className="image-preview" />}
      </header>
    </div>
  );
}

export default App;