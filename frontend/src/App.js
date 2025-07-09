import React from "react";
import MicrobiomeBuilder from "./components/MicrobiomeBuilder";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ReactorSetup from "./components/ReactorSetup";
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MicrobiomeBuilder />} />
        <Route path="/reactor" element={<ReactorSetup />} />
      </Routes>
    </Router>
  );
}

export default App;
