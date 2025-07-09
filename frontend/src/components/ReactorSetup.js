import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useLocation } from "react-router-dom";


export default function ReactorSetup() {
  const [media, setMedia] = useState([]);
  const [subpopulations, setSubpopulations] = useState([]);
  const navigate = useNavigate();
  const location = useLocation();
  const setup = location.state?.setup || "Kombucha"; // fallback to "Kombucha" if not passed


  useEffect(() => {
    fetchMedia();
    fetchSubpopulations();
  }, []);

  const fetchMedia = async () => {
    try {
      const response = await axios.get("http://localhost:5000/api/get_media");
      setMedia(response.data);
    } catch (error) {
      console.error("Error fetching media:", error);
    }
  };

  const fetchSubpopulations = async () => {
    try {
      const response = await axios.get("http://localhost:5000/api/get_subpopulations");
      setSubpopulations(response.data);
    } catch (error) {
      console.error("Error fetching subpopulations:", error);
    }
  };

  const handleMediaChange = (index, value) => {
    const updated = [...media];
    updated[index].concentration = value;
    setMedia(updated);
  };

  const handleSubpopChange = (index, value) => {
    const updated = [...subpopulations];
    updated[index].count = value;
    setSubpopulations(updated);
  };

  const handleSubmit = async () => {
    try {
      await axios.post("http://localhost:5000/api/update_reactor", {
        media,
        subpopulations,
      });
      alert("Reactor parameters saved successfully!");
      const encodedSetup = encodeURIComponent(setup);
      window.location.href = `http://localhost:8501/?setup=${encodedSetup}`;
    } catch (error) {
      console.error("Error updating reactor:", error);
      alert("Failed to save reactor parameters.");
    }
  };

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Create Reactor</h1>

      {/* Media */}
      <h2 className="text-xl font-semibold mb-2">Media</h2>
      {media.map((item, index) => (
        <div key={index} className="flex gap-2 mb-2">
          <span className="flex-1">{item.metabolite}</span>
          <input
            className="border p-2 flex-1"
            placeholder="Concentration"
            value={item.concentration}
            onChange={(e) => handleMediaChange(index, e.target.value)}
          />
        </div>
      ))}

      {/* Subpopulations */}
      <h2 className="text-xl font-semibold mt-4 mb-2">Subpopulations</h2>
      {subpopulations.map((sub, index) => (
        <div key={index} className="flex gap-2 mb-2">
          <span className="flex-1">{sub.id}</span>
          <input
            className="border p-2 flex-1"
            placeholder="Count"
            value={sub.count}
            onChange={(e) => handleSubpopChange(index, e.target.value)}
          />
        </div>
      ))}

      <button
        className="bg-green-600 text-white px-4 py-2 rounded mt-4"
        onClick={handleSubmit}
      >
        Save Reactor
      </button>
    </div>
  );
}
