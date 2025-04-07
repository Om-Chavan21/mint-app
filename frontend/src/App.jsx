// App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const mintInfoData = {
  "apple_mint": {
    name: "Apple Mint (Mentha suaveolens)",
    description: "Apple mint has a fruity scent that is reminiscent of apples. It has round, woolly leaves and is used in teas, desserts, and as a garnish.",
    uses: "Teas, cocktails, salads, desserts, jellies"
  },
  "aquatic_mint": {
    name: "Aquatic Mint",
    description: "Aquatic mint grows in moist, wet areas and has strongly scented leaves. It's often found near water bodies.",
    uses: "Medicinal teas, aromatic gardens"
  },
  "chocolate_mint": {
    name: "Chocolate Mint (Mentha x piperita 'Chocolate')",
    description: "Chocolate mint has a distinct chocolate-mint aroma with dark stems and green leaves. It's a popular culinary herb.",
    uses: "Desserts, hot chocolate, ice cream, teas"
  },
  "mexican_mint": {
    name: "Mexican Mint (Plectranthus amboinicus)",
    description: "Mexican mint (also called Cuban oregano) has thick, fuzzy leaves with a strong oregano-like scent.",
    uses: "Mexican and Caribbean cooking, stews, bean dishes"
  },
  "mojito_mint": {
    name: "Mojito Mint (Mentha x villosa)",
    description: "Mojito mint is the traditional mint used in Cuban mojitos. It has a warm, aromatic flavor less intense than spearmint.",
    uses: "Mojito cocktails, Cuban cuisine, fruit salads"
  },
  "peppermint": {
    name: "Peppermint (Mentha × piperita)",
    description: "Peppermint has a strong, cool, menthol flavor with purple/green pointed leaves. It's one of the most common mint varieties.",
    uses: "Teas, desserts, candies, essential oils, digestive remedies"
  },
  "pineapple_mint": {
    name: "Pineapple Mint (Mentha suaveolens 'Variegata')",
    description: "Pineapple mint features variegated leaves with white edges and a fruity scent reminiscent of pineapple.",
    uses: "Fruit salads, garnishes, infused water, decorative gardens"
  },
  "spearmint": {
    name: "Spearmint (Mentha spicata)",
    description: "Spearmint has bright green pointed leaves with a sweet, mild flavor. It's the most common culinary mint variety.",
    uses: "Middle Eastern cuisine, teas, cocktails, jellies, sauces"
  },
  "non_mint": {
    name: "Not a Mint Leaf",
    description: "This doesn't appear to be a mint variety. While it might be another herb or plant, it lacks the distinctive characteristics of the mint family.",
    uses: "N/A"
  }
};

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('resnet18');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setResults(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(
        `https://3baf-34-125-159-189.ngrok-free.app/classify/?model_name=${selectedModel}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      setResults(response.data);
    } catch (error) {
      console.error('Error classifying image:', error);
      alert('Error classifying image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  // Get the top prediction class for displaying detailed information
  const topPrediction = results?.predictions?.[0]?.class || '';
  const mintInfo = mintInfoData[topPrediction] || mintInfoData["non_mint"];

  return (
    <div className="App">
      <header>
        <h1>Mint Leaf Classifier</h1>
        <p>Upload an image to identify different types of mint leaves</p>
      </header>

      <div className="container">
        <div className="upload-section">
          <div className="model-selection">
            <label>Select Model:</label>
            <select value={selectedModel} onChange={handleModelChange}>
              <option value="resnet18">ResNet18</option>
              <option value="mobilenet_v2">MobileNet v2</option>
              <option value="efficientnet_b0">EfficientNet B0</option>
              <option value="densenet121">DenseNet121</option>
            </select>
          </div>

          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange}
            id="file-input"
            className="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            Choose an image
          </label>

          {preview && (
            <div className="image-preview">
              <img src={preview} alt="Preview" />
              <button 
                onClick={handleSubmit} 
                disabled={loading}
                className="classify-button"
              >
                {loading ? 'Classifying...' : 'Classify'}
              </button>
            </div>
          )}
        </div>

        {results && (
          <div className="results-section">
            <h2>Classification Results</h2>
            <div className="prediction-bar-container">
              {results.predictions.map((pred, index) => (
                <div key={index} className="prediction-item">
                  <div className="prediction-label">
                    {pred.class === 'non_mint' ? 'Not a Mint Leaf' : pred.class.replace('_', ' ')}
                  </div>
                  <div className="prediction-bar-wrapper">
                    <div 
                      className="prediction-bar" 
                      style={{ width: `${pred.probability}%` }}
                    ></div>
                    <span className="probability">{pred.probability}%</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="mint-info">
              <h3>{mintInfo.name}</h3>
              <p>{mintInfo.description}</p>
              <p><strong>Common Uses:</strong> {mintInfo.uses}</p>
              <div className="mint-status">
                {results.is_mint ? (
                  <span className="is-mint">✓ This is a mint leaf</span>
                ) : (
                  <span className="not-mint">✗ This is not a mint leaf</span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;