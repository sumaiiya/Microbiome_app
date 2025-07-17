import React, { useState } from "react";
import "./MicrobiomeBuilder.css";
import { useNavigate } from "react-router-dom";

export default function MicrobiomeBuilder() {

	const navigate = useNavigate();
	const [model, setModel] = useState("");
	// State for environment setup, defaulting to "Kombucha"
	const [setup, setSetup] = useState("Kombucha");
	const [species, setSpecies] = useState([]);
	const [media, setMedia] = useState([]);
	const [metabolites, setMetabolites] = useState([]);
	const [transitions, setTransitions] = useState([]);

	const handleClearMicrobiome = async () => {
		if (!window.confirm("Are you sure you want to clear all microbiome data? This cannot be undone.")) {
			return;
		}

		try {
			const response = await fetch("http://localhost:5000/api/clear_microbiome", {
				method: "POST",
			});
			const result = await response.json();
			if (result.status === "success") {
				alert("Microbiome data cleared on server.");
				// Clear local frontend state as well:
				setSpecies([]);
				setMedia([]);
				setMetabolites([]);
				setModel("");
				setSetup("Kombucha"); // or empty string if you prefer
			} else {
				alert("Error clearing data: " + result.message);
			}
		} catch (err) {
			alert("Network error: " + err.message);
		}
	};
	const handleAddSpecies = () => {
		setSpecies([...species, {
			id: "", name: "", color: "#ff6600", subpopulations: [],
		}]);
	};

	const handleUpdateSpecies = (index, field, value) => {
		const updated = [...species];
		updated[index][field] = value;
		setSpecies(updated);
	};

	const handleAddSubpop = (speciesIndex) => {
		const updated = [...species];
		updated[speciesIndex].subpopulations.push({
			id: "", count: "", mumax: "", pHopt: "", pHalpha: "", feeding: [],
		});
		setSpecies(updated);
	};

	const handleRemoveSubpop = (speciesIndex, subIndex) => {
		const updated = [...species];
		updated[speciesIndex].subpopulations.splice(subIndex, 1);
		setSpecies(updated);
	};

	const handleUpdateSubpop = (speciesIndex, subIndex, field, value) => {
		const updated = [...species];
		updated[speciesIndex].subpopulations[subIndex][field] = value;
		setSpecies(updated);
	};

	const handleAddFeedingTerm = (speciesIndex, subIndex) => {
		const updated = [...species];
		updated[speciesIndex].subpopulations[subIndex].feeding.push({
			name: "", in: [], out: [],
		});
		setSpecies(updated);
	};

	const handleRemoveFeedingTerm = (speciesIndex, subIndex, termIndex) => {
		const updated = [...species];
		updated[speciesIndex].subpopulations[subIndex].feeding.splice(termIndex, 1);
		setSpecies(updated);
	};

	const handleUpdateFeedingTerm = (speciesIndex, subIndex, termIndex, field, value) => {
		const updated = [...species];
		updated[speciesIndex].subpopulations[subIndex].feeding[termIndex][field] = value;
		setSpecies(updated);
	};

	const handleAddInOut = (speciesIndex, subIndex, termIndex, type) => {
		const updated = [...species];
		updated[speciesIndex].subpopulations[subIndex].feeding[termIndex][type].push({
			species: "", yield: "", monodK: "",
		});
		setSpecies(updated);
	};

	const handleUpdateInOut = (speciesIndex, subIndex, termIndex, type, itemIndex, field, value) => {
		const updated = [...species];
		let newValue = value;

		if (field === "yield") {
			const numericValue = parseFloat(value) || 0;
			newValue = type === "in" ? -Math.abs(numericValue) : Math.abs(numericValue); // Force sign
		}

		if (field === "monodK" && type === "out") {
			newValue = 0;
		}

		updated[speciesIndex].subpopulations[subIndex].feeding[termIndex][type][itemIndex][field] = newValue;
		setSpecies(updated);
	};

	const handleRemoveInOut = (speciesIndex, subIndex, termIndex, type, itemIndex) => {
		const updated = [...species];
		updated[speciesIndex].subpopulations[subIndex].feeding[termIndex][type].splice(itemIndex, 1);
		setSpecies(updated);
	};

	const handleAddMedia = () => {
		setMedia([...media, { metabolite: "", concentration: "" }]);
	};

	const handleUpdateMedia = (index, field, value) => {
		const updated = [...media];
		updated[index][field] = value;
		setMedia(updated);
	};

	const handleRemoveMedia = (index) => {
		setMedia(media.filter((_, i) => i !== index));
	};

	const handleAddMetabolite = () => {
		setMetabolites([...metabolites, { id: "", color: "#000000", MolecularWeight: "" }]);
	};

	const handleUpdateMetabolite = (index, field, value) => {
		const updated = [...metabolites];
		updated[index][field] = value;
		setMetabolites(updated);
	};

	const handleRemoveMetabolite = (index) => {
		setMetabolites(metabolites.filter((_, i) => i !== index));
	};

	const handleSubmit = async () => {
		const reactorConfig = { model, setup, species, media, metabolites, transitions };
		try {
			const response = await fetch("http://localhost:5000/api/save_microbiome", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(reactorConfig),
			});
			const result = await response.json();
			alert("Success: " + result.message);
			// 🔀 Navigate to the reactor page
			//navigate("/reactor");
			navigate("/reactor", { state: { setup } });

			const encodedSetup = encodeURIComponent(setup);
			//window.location.href = `http://localhost:8501/?setup=${encodedSetup}`;
		} catch (err) {
			console.error("Failed to save data:", err);
			alert("Failed to save data.");

		}
	};

	return (
		<div className="app-wrapper">

			<div className="microbiome-container">
				<button
					className="bg-red-700 text-white px-6 py-3 rounded mt-4 mb-6"
					onClick={handleClearMicrobiome}
				>
					Clear Data
				</button>
				<h1 className="text-2xl font-bold">Microbiome Environment Builder</h1>

				{/* Model + Setup */}
				<div>
					<label className="block mb-1 font-semibold">Select Model:</label>
					<select
						className="border p-2 w-1/2 mb-4"
						value={model}
						onChange={(e) => setModel(e.target.value)}
					>
						<option value="">-- Select Model --</option>
						<option value="Kinetic">Kinetic</option>
						<option value="Stochastic">Stochastic</option>
					</select>
					<div>
						<label className="block mb-1 font-semibold">Enter Environment Type:</label>
						<input
							type="text"
							className="border p-2"
							style={{ width: "200px" }}
							value={setup}
							onChange={(e) => setSetup(e.target.value)}
						/>
					</div>

				</div>

				{/* Species */}
				<div>
					<h2 className="text-xl font-semibold mb-2">Species</h2>
					{species.map((sp, sIndex) => (
						<div key={sIndex} className="border p-3 mb-3 rounded">
							<input className="border p-2 w-full mb-2" placeholder="Species ID" value={sp.id}
								onChange={(e) => handleUpdateSpecies(sIndex, "id", e.target.value)} />
							<input className="border p-2 w-full mb-2" placeholder="Species Name" value={sp.name}
								onChange={(e) => handleUpdateSpecies(sIndex, "name", e.target.value)} />
							<input type="color" className="mb-2" value={sp.color}
								onChange={(e) => handleUpdateSpecies(sIndex, "color", e.target.value)} />

							{/* Subpopulations */}
							<h3 className="mt-2 font-medium">Subpopulations</h3>
							{sp.subpopulations.map((sub, subIndex) => (
								<div key={subIndex} className="bg-gray-100 p-3 mt-2 rounded">
									<div className="flex justify-between items-center mb-2">
										<h4 className="font-semibold">Subpopulation #{subIndex + 1}</h4>
										<button
											className="bg-red-600 text-white px-3 py-1 rounded"
											onClick={() => {
												const updated = [...species];
												updated[sIndex].subpopulations.splice(subIndex, 1);
												setSpecies(updated);
											}}
										>
											Remove Subpopulation
										</button>
									</div>
									<input className="border p-1 mb-1 w-full" placeholder="Subpopulation id"
										value={sub.name} onChange={(e) => handleUpdateSubpop(sIndex, subIndex, "id", e.target.value)} />
									<input className="border p-1 mb-1 w-full" placeholder="Count"
										value={sub.count || 0} disabled />
									<input className="border p-1 mb-1 w-full" placeholder="mumax"
										value={sub.mumax} onChange={(e) => handleUpdateSubpop(sIndex, subIndex, "mumax", e.target.value)} />
									<input className="border p-1 mb-1 w-full" placeholder="pHopt"
										value={sub.pHopt} onChange={(e) => handleUpdateSubpop(sIndex, subIndex, "pHopt", e.target.value)} />
									<input className="border p-1 mb-1 w-full" placeholder="pHalpha"
										value={sub.pHalpha} onChange={(e) => handleUpdateSubpop(sIndex, subIndex, "pHalpha", e.target.value)} />
									<select
										className="border p-1 mb-1 w-full"
										value={sub.state || "active"}
										onChange={(e) => handleUpdateSubpop(sIndex, subIndex, "state", e.target.value)}
									>
										<option value="active">Active</option>
										<option value="inactive">Inactive</option>

									</select>

									{/* Feeding terms */}
									<div className="mt-3">
										<h4 className="font-semibold mb-1">Feeding Terms</h4>
										{sub.feeding.map((term, termIndex) => (
											<div key={termIndex} className="feeding-card">
												<input className="border p-2 w-full mb-2" placeholder="Feeding Term Name"
													value={term.name}
													onChange={(e) =>
														handleUpdateFeedingTerm(sIndex, subIndex, termIndex, "name", e.target.value)
													} />

												{/* Consumed */}
												<h5 className="font-medium mb-1">Consume</h5>
												{term.in.map((item, i) => (
													<div key={i} className="inout-row">
														<input className="border p-2 flex-1" placeholder="Metabolite"
															value={item.metabolite}
															onChange={(e) => handleUpdateInOut(sIndex, subIndex, termIndex, "in", i, "metabolite", e.target.value)} />
														<input className="border p-2 flex-1" placeholder="Yield"
															value={item.yield}
															onChange={(e) => handleUpdateInOut(sIndex, subIndex, termIndex, "in", i, "yield", e.target.value)} />
														<input className="border p-2 flex-1" placeholder="MonodK"
															value={item.monodK}
															onChange={(e) => handleUpdateInOut(sIndex, subIndex, termIndex, "in", i, "monodK", e.target.value)} />
														<button className="bg-red-600 text-white px-2 py-1 rounded"
															onClick={() => handleRemoveInOut(sIndex, subIndex, termIndex, "in", i)}>Remove</button>
													</div>
												))}
												<button className="bg-blue-600 text-white px-2 py-1 rounded"
													onClick={() => handleAddInOut(sIndex, subIndex, termIndex, "in")}>+ Add In</button>

												{/* Produced */}
												<h5 className="font-medium mt-2 mb-1">Produced / OUT</h5>
												{term.out.map((item, i) => (
													<div key={i} className="inout-row">
														<input className="border p-2 flex-1" placeholder="Metabolite"
															value={item.metabolite}
															onChange={(e) => handleUpdateInOut(sIndex, subIndex, termIndex, "out", i, "metabolite", e.target.value)} />
														<input className="border p-2 flex-1" placeholder="Yield"
															value={item.yield}
															onChange={(e) => handleUpdateInOut(sIndex, subIndex, termIndex, "out", i, "yield", e.target.value)} />
														<input className="border p-2 flex-1 bg-gray-200 cursor-not-allowed" placeholder="MonodK"
															value={0} disabled />
														<button className="bg-red-600 text-white px-2 py-1 rounded"
															onClick={() => handleRemoveInOut(sIndex, subIndex, termIndex, "out", i)}>Remove</button>
													</div>
												))}
												<button className="bg-blue-600 text-white px-2 py-1 rounded"
													onClick={() => handleAddInOut(sIndex, subIndex, termIndex, "out")}>+ Add Out</button>

												<button className="bg-red-600 text-white px-3 py-1 rounded mt-2"
													onClick={() => handleRemoveFeedingTerm(sIndex, subIndex, termIndex)}>Remove Feeding Term</button>
											</div>
										))}

										<button className="bg-green-600 text-white px-4 py-2 rounded"
											onClick={() => handleAddFeedingTerm(sIndex, subIndex)}>+ Add Feeding Term</button>
									</div>
								</div>
							))}
							<button className="bg-blue-500 text-white px-3 py-1 mt-2 rounded"
								onClick={() => handleAddSubpop(sIndex)}>+ Add Subpopulation</button>
						</div>
					))}
					<button className="bg-green-600 text-white px-4 py-2 rounded"
						onClick={handleAddSpecies}>+ Add Species</button>
				</div>
				<div className="mt-10">
					<h2 className="text-xl font-semibold mb-2">Transitions Between Subpopulations</h2>
					{transitions.map((tr, index) => (
						<div key={index} className="border p-3 mb-2 rounded bg-gray-50">
							<input className="border p-2 mr-2" placeholder="From Subpopulation ID"
								value={tr.subpopulation_A}
								onChange={(e) => {
									const updated = [...transitions];
									updated[index].subpopulation_A = e.target.value;
									setTransitions(updated);
								}} />
							<input className="border p-2 mr-2" placeholder="To Subpopulation ID"
								value={tr.subpopulation_B}
								onChange={(e) => {
									const updated = [...transitions];
									updated[index].subpopulation_B = e.target.value;
									setTransitions(updated);
								}} />
							<input className="border p-2 mr-2 w-1/2" placeholder="Rule"
								value={tr.hillFunc}
								onChange={(e) => {
									const updated = [...transitions];
									updated[index].hillFunc = e.target.value;
									setTransitions(updated);
								}} />
							<input className="border p-2 mr-2 w-28" placeholder="Rate"
								value={tr.rate}
								onChange={(e) => {
									const updated = [...transitions];
									updated[index].rate = e.target.value;
									setTransitions(updated);
								}} />
							<button className="bg-red-500 text-white px-3 py-1 rounded"
								onClick={() => {
									const updated = transitions.filter((_, i) => i !== index);
									setTransitions(updated);
								}}>Remove</button>
						</div>
					))}
					<button className="bg-green-700 text-white px-4 py-2 rounded"
						onClick={() => setTransitions([...transitions, {
							subpopulation_A: "", subpopulation_B: "", hillFunc: "", rate: 0.01
						}])}>
						+ Add Transition
					</button>
				</div>

				<button className="bg-black text-white px-6 py-3 rounded mt-6"
					onClick={handleSubmit}>Save Microbiome</button>
			</div>
		</div>
	);
}
