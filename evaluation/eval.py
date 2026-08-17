from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import json_repair
import numpy as np
from tqdm import tqdm

from vllm import LLM, SamplingParams

from prompts import (
    Output,
    prompt_m1_concept_node_validity_ordinal,
    prompt_m1_concept_triplet_accuracy_ordinal,
)

# --------------------------------------------------
# vLLM structured outputs compatibility layer
# --------------------------------------------------
# Newer vLLM: StructuredOutputsParams + SamplingParams(structured_outputs=...)
# Older vLLM: GuidedDecodingParams + SamplingParams(guided_decoding=...)
try:
    from vllm.sampling_params import StructuredOutputsParams  # vLLM newer API
except Exception:
    StructuredOutputsParams = None  # type: ignore

try:
    from vllm.sampling_params import GuidedDecodingParams  # older API (removed in v0.12+)
except Exception:
    GuidedDecodingParams = None  # type: ignore


def _make_sampling_params(schema: dict, temperature: float, max_tokens: int) -> SamplingParams:
    """
    Build SamplingParams with JSON-structured decoding if supported by the installed vLLM.
    Falls back gracefully if the installed version doesn't support the chosen fields.
    """
    base_kwargs = dict(temperature=temperature, max_tokens=max_tokens)

    # 1) New vLLM path: structured_outputs
    if StructuredOutputsParams is not None:
        try:
            so = StructuredOutputsParams(json=schema)
            return SamplingParams(**base_kwargs, structured_outputs=so)
        except TypeError:
            # SamplingParams doesn't accept structured_outputs in this version
            pass
        except Exception:
            pass

    # 2) Old vLLM path: guided_decoding
    if GuidedDecodingParams is not None:
        try:
            gd = GuidedDecodingParams(json=schema)
            return SamplingParams(**base_kwargs, guided_decoding=gd)
        except TypeError:
            # SamplingParams doesn't accept guided_decoding in this version
            pass
        except Exception:
            pass

    # 3) Fallback: no structured decoding
    return SamplingParams(**base_kwargs)


# --------------------------------------------------
# LLM batch inference
# --------------------------------------------------
def batch_llm_inference(llm: LLM, messages_list: List[List[Dict]], schema: dict,
                        temperature: float = 0.0, max_tokens: int = 2048) -> List[Optional[dict]]:
    params = _make_sampling_params(schema=schema, temperature=temperature, max_tokens=max_tokens)

    # Use keyword argument for compatibility across vLLM versions
    raw = llm.chat(messages_list, sampling_params=params, use_tqdm=False)

    outputs: List[Optional[dict]] = []
    for r in raw:
        # vLLM returns objects with .outputs[0].text in offline mode
        text = r.outputs[0].text
        try:
            json_output = json_repair.loads(text)
            if (isinstance(json_output, list)) and json_output and isinstance(json_output[-1], dict):
                json_output = json_output[-1]
            outputs.append(json_output if isinstance(json_output, dict) else None)
        except Exception as e:
            print(f"⚠ JSON parsing error: {e}")
            outputs.append(None)

    return outputs


# --------------------------------------------------
# Metric evaluators
# --------------------------------------------------
def eval_node_significance(llm: LLM, data: List[dict], course_name: str):
    node_to_excerpts: Dict[str, List[str]] = {}

    for item in data:
        if item.get("relation") is None:
            continue
        for side in ["A", "B"]:
            node = item[side]["name"]
            excerpts = [e["text"] for e in item.get("evidence_chunks", []) if "text" in e]
            node_to_excerpts.setdefault(node, []).extend(excerpts)

    prompts = [
        [{
            "role": "user",
            "content": prompt_m1_concept_node_validity_ordinal(node, ex[:5], course_name)
        }]
        for node, ex in node_to_excerpts.items()
    ]
    print(f"Generated {len(prompts)} prompts for node significance")

    outputs = batch_llm_inference(llm, prompts, Output.model_json_schema())
    if outputs:
        print(f"First output: {outputs[0]}")

    scores = []
    for o in outputs:
        if isinstance(o, dict) and "score" in o:
            try:
                scores.append(float(o["score"]))
            except Exception:
                pass

    if scores:
        return {
            "mean": float(np.mean(scores)) / 2.0,
            "std": float(np.std(scores)) / 2.0
        }
    return None


def eval_triplet_accuracy(llm: LLM, data: List[dict], course_name: str):
    prompts = []

    for item in data:
        edge = {
            "source": item["A"]["name"],
            "relation_type": "None" if item.get("relation") is None else item["relation"],
            "target": item["B"]["name"],
        }
        excerpts = [e["text"] for e in item.get("evidence_chunks", []) if "text" in e]

        prompts.append([{
            "role": "user",
            "content": prompt_m1_concept_triplet_accuracy_ordinal(edge, excerpts[:5], course_name)
        }])

    print(f"Generated {len(prompts)} prompts for triplet accuracy")
    outputs = batch_llm_inference(llm, prompts, Output.model_json_schema())
    if outputs:
        print(f"First output: {outputs[0]}")

    scores = []
    for o in outputs:
        if isinstance(o, dict) and "score" in o:
            try:
                scores.append(float(o["score"]))
            except Exception:
                pass

    if scores:
        return {
            "mean": float(np.mean(scores)) / 2.0,
            "std": float(np.std(scores)) / 2.0
        }
    return None


# --------------------------------------------------
# Process single file
# --------------------------------------------------
def process_file(llm: LLM, path: str, course_name: str, method_name: str):
    """Process a single JSONL file and return results"""
    fname = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")
    print(f"Method: {method_name}")
    print(f"Course: {course_name}")
    print(f"{'='*60}")

    # Load and parse JSONL with better error handling
    data: List[dict] = []
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            print(f"File has {len(lines)} lines")

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠ JSON error on line {i+1}: {e}")
                    continue

            print(f"Successfully parsed {len(data)} valid JSON objects")

    except FileNotFoundError:
        print(f"✗ File not found: {path}")
        return None
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return None

    if not data:
        print("✗ No valid data loaded - file may be empty or malformed")
        return None

    # Run evaluations
    try:
        print("\n→ Evaluating node significance...")
        ns = eval_node_significance(llm, data, course_name)
        print(f"  Result: {ns}")

        print("\n→ Evaluating triplet accuracy...")
        ta = eval_triplet_accuracy(llm, data, course_name)
        print(f"  Result: {ta}")

        return {"node_significance": ns, "triplet_accuracy": ta}

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# --------------------------------------------------
# Main evaluation loop
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        default="experiments_outputs",
        help="Root directory containing method folders with JSONL files",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="Single JSONL file to evaluate (for testing)",
    )
    parser.add_argument(
        "--course_name",
        default=None,
        help="Course name (required when using --input_file). Options: 'algo', 'anlp', 'sql'",
    )
    parser.add_argument(
        "--method_name",
        default="test_method",
        help="Method name for single file evaluation (default: 'test_method')",
    )
    parser.add_argument("--output_json", default="final_eval_complete.json")
    parser.add_argument("--model_name", default="openai/gpt-oss-120b")
    # [JR] Was hardcoded to 131072, which fails on any shorter-context model.
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="vLLM context length. Default: read from the model's config.",
    )
    args = parser.parse_args()

    print(f"Initializing vLLM with model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_seqs=400,
        trust_remote_code=True,
    )
    print("✓ Model loaded")

    # SINGLE FILE MODE
    if args.input_file:
        if not args.course_name:
            print("✗ Error: --course_name required when using --input_file")
            print("   Options: 'algo', 'anlp', 'sql'")
            return

        ican_scope = (
            "an accelerated discrete mathematics and algorithms sequence for graduate "
            "students entering computer science from non-CS backgrounds, covering "
            "counting and combinatorics, proofs and propositional logic, recursion and "
            "solving recurrences, asymptotic analysis, core data structures, graph "
            "traversals and shortest paths, divide-and-conquer, greedy algorithms, "
            "dynamic programming, finite automata and Turing machines, and P vs NP with "
            "polynomial-time reductions and NP-completeness"
        )
        # [JR] official UIUC course catalog text
        me200_catalog_scope = (
            "introduction to classical thermodynamics through the second law; system "
            "and control volume analyses of thermodynamic processes; irreversibility "
            "and availability; relations for ideal gas mixtures. Topics: definitions, "
            "properties, equations of state, state postulate, compressibility charts; "
            "work, heat transfer, and the first law of thermodynamics for a control "
            "mass (closed system); the first law for a control volume (open system), "
            "steady state and unsteady analysis; entropy, the second law of "
            "thermodynamics, component efficiency, cycle efficiency (power and "
            "refrigeration cycles); reversible work, availability, irreversibility; "
            "properties and thermodynamics of ideal gas mixtures"
        )
        me400_catalog_scope = (
            "processes and systems for energy conversion, including power and "
            "refrigeration cycles, air conditioning, thermoelectrics, and fuel cells; "
            "ideal gas mixtures and psychrometrics. Topics: introduction and review of "
            "thermodynamics; chemical reactions; power cycles; refrigeration cycles; "
            "air-conditioning; direct energy conversion"
        )
        tam210_catalog_scope = (
            "forces, moments, couples; resultants of force systems; equilibrium "
            "analysis and free-body diagrams; analysis of forces acting on members "
            "of trusses, frames, etc.; shear-force and bending-moment distributions; "
            "Coulomb friction; centroids and center of mass; applications of statics "
            "in design. Topics: review of vector algebra; forces, moments, couples; "
            "equilibrium, equipollent systems, resultants, distributed forces; "
            "equilibrium analysis, free-body diagrams, practical examples; trusses, "
            "methods of joint and sections; multi-force members, shear-force and "
            "bending-moment diagrams; statics and structural design; Coulomb "
            "friction, applications; centroids and center of mass"
        )
        tam212_catalog_scope = (
            "kinematics and dynamics of the three-dimensional motion of particles; "
            "kinematics and dynamics of the plane motion of rigid bodies; methods of "
            "work energy and impulse momentum; moving reference frames. Topics: "
            "kinematics of a particle and of a system of particles; dynamics of "
            "particles, Newton's laws, applications; kinematics of 2D motion of rigid "
            "bodies; dynamics of 2D motion of rigid bodies; methods of work-energy "
            "and impulse-momentum; moving reference frames"
        )
        me270_catalog_scope = (
            "introduction to DFM methodologies and tools; material selection (new and "
            "traditional materials); designing for primary manufacturing processes "
            "(cutting fundamentals, casting, forming, and shaping); designing with "
            "plastics (snap-fits, integral hinges, etc.); design for assembly (DFA); "
            "geometric dimensioning and tolerancing (GD&T). Topics: DFM overview and "
            "strategy; quality function deployment (QFD); concept selection; product "
            "design specification; design for assembly; design for economic manufacture "
            "(material removal, casting, forming, and shaping); designing with plastics; "
            "geometric dimensioning and tolerancing; selection of materials (life-cycle "
            "economics). Laboratory topics: machine tools; injection molding; CNC "
            "machining; sand casting; rapid prototyping; water jet cutting; design of "
            "experiments; inspection and metrology (CMM/GD&T)"
        )
        me310_catalog_scope = (
            "introduction to fluid mechanics with coverage of theory and applications of "
            "incompressible viscous and inviscid flows, and compressible high speed "
            "flows. Topics: definitions and fluid kinematics; hydrostatics, including "
            "manometers, Bourdon gauges and pressure transducers; control volume "
            "equations for continuity, linear momentum, angular momentum, and energy; "
            "Bernoulli's equation, including pitot tubes, Venturi meters, orifice meters "
            "and hot wire anemometers; differential continuity and momentum equations; "
            "dimensional similitude and model testing; viscous flows and pipe flows; "
            "boundary layers, lift and drag; potential flow, superposition, numerical "
            "solutions; compressible flow. Laboratory topics: fluid properties; "
            "centrifugal pump characterization; free air jet; pipe flow; cylinder in "
            "cross flow"
        )
        me320_catalog_scope = (
            "principles and application of heat transfer by conduction, convection, and "
            "thermal radiation. Topics: modes of heat transfer; temperature and "
            "measurement devices; steady-state one-dimensional heat conduction; extended "
            "surface heat transfer; transient one-dimensional heat conduction; numerical "
            "methods in conduction; radiation heat transfer; wavelength-dependent surface "
            "properties; directional characteristics of thermal radiation; view factors; "
            "graybody exchange; convection heat transfer; external and internal flows; "
            "heat exchangers. Laboratory topics: temperature measurement; conduction; "
            "convection; heat exchangers; radiation"
        )
        me340_catalog_scope = (
            "dynamic modeling of mechanical components and systems; time domain and "
            "frequency domain analysis of linear time invariant systems; multi-degree of "
            "freedom systems; linearization of nonlinear systems. Topics: Laplace "
            "transformation, inverse transformation, solution of differential equations, "
            "transfer functions, poles and zeros; modeling of dynamic systems by "
            "conservation principles for mass, energy, fluid flow, heat transfer, and "
            "mechanical and electromechanical systems, state (phase) space "
            "representation; dynamic system classification, linearization of nonlinear "
            "systems, dynamic simulation; time domain analysis of linear time invariant "
            "systems, first and second order systems, time constant, damping ratio and "
            "natural frequency, impulse response and convolution integral; frequency "
            "domain analysis, frequency response, vibration isolation, base excitation, "
            "measurement systems, Fourier series analysis; multi-degree-of-freedom "
            "systems, natural frequencies and normal modes, beat generation and vibration "
            "absorbers. Laboratory topics: complex numbers, partial fractions, "
            "eigenvalues and eigenvectors; first-order systems and system identification; "
            "block diagrams, transfer functions and simulation; second-order systems, "
            "damping regimes; mode shapes and resonance; continuous systems and beam "
            "vibration; nonlinear systems, Lagrange's equations, equilibria and stability"
        )
        tam251_catalog_scope = (
            "relationship between internal stresses and deformations produced by external "
            "forces acting on deformable bodies, and design principles based on mechanics "
            "of solids: normal stresses, shear stresses, and deformations produced by "
            "tensile, compressive, torsional, and bending loading of members; beam "
            "deflections; elastic energy and impact; multi-dimensional stress states; and "
            "buckling of columns. Topics: basic concepts of stress and strain; uniaxial "
            "loading and deformation, statically determinate and indeterminate problems, "
            "design based on yield strength and ultimate strength; torsion of circular "
            "shafts and thin-walled sections, geometry of deformation, stress "
            "distribution, design of shafts for power transmission; stresses due to "
            "bending, symmetric elastic beams, transverse shear, built-up beams, design of "
            "beams for structural applications; beam deflections, double integration, "
            "direct integration, method of superposition; multi-axial stress and strain "
            "states, transformation of stress and strain, Mohr's circle, principal "
            "stresses and strains, plane stress and plane strain, yield criteria; buckling "
            "of columns, Euler theory"
        )
        mse280_catalog_scope = (
            "materials science and engineering of ceramics, electronic materials, metals "
            "and polymers; bonding; crystallography; imperfections; processing and "
            "properties of semiconductors, polymers, metals, ceramics and composites; "
            "phase diagrams; case studies"
        )
        mse494_catalog_scope = (
            "introduction to design methodologies in the context of materials science and "
            "engineering. Topics: human centered design (HCD); statistical modeling; "
            "design tradeoffs; material selection; materials design; team management; "
            "development of design projects for implementation in a subsequent course "
            "(MSE 495); objectives and constraints such as economic, manufacturability, "
            "environmental, ethical, health and safety, sustainability, social, and "
            "political concerns as they relate to project design"
        )
        mse495_catalog_scope = (
            "continuation of MSE 494: design teams evaluate alternatives, finalize "
            "concepts, model and analyze solutions, build and test a final product "
            "(physical or digital), and present the results professionally; solutions "
            "build on earlier course work and incorporate realistic constraints"
        )
        # [JR] me200/me400 keep the bare-title vs `_catalog` split that the completed
        # 2x2 and 2x3 comparisons used. Courses added after that comparison carry the
        # scope string on the bare key, since the bare-title condition was retired.
        course_map = {
            "algo": "Efficient Algorithms and Intractable Problems",
            "anlp": "Advanced Topics in Natural Language Processing",
            "sql": "Database Systems",
            "me200": "Thermodynamics",
            "me200_catalog": f"Thermodynamics (ME 200) — {me200_catalog_scope}",
            "me400": "Energy Conversion Systems",
            "me400_catalog": f"Energy Conversion Systems (ME 400) — {me400_catalog_scope}",
            "cs401": f"iCAN Algorithms (CS 401) — {ican_scope}",
            "cs403": f"iCAN Algorithms (CS 403) — {ican_scope}",
            "cs401_403": f"iCAN Algorithms (CS 401/403) — {ican_scope}",
            "tam210": f"Statics (TAM 210) — {tam210_catalog_scope}",
            "tam212": f"Introductory Dynamics (TAM 212) — {tam212_catalog_scope}",
            "tam251": f"Introductory Solid Mechanics (TAM 251) — {tam251_catalog_scope}",
            "me270": f"Design for Manufacturability (ME 270) — {me270_catalog_scope}",
            "me310": f"Fundamentals of Fluid Dynamics (ME 310) — {me310_catalog_scope}",
            "me320": f"Heat Transfer (ME 320) — {me320_catalog_scope}",
            "me340": f"Dynamics of Mechanical Systems (ME 340) — {me340_catalog_scope}",
            "mse280": f"Engineering Materials (MSE 280) — {mse280_catalog_scope}",
            "mse494": f"Materials Design Thinking (MSE 494) — {mse494_catalog_scope}",
            "mse495": f"Materials Design (MSE 495) — {mse495_catalog_scope}",
        }
        if args.course_name not in course_map:
            print(f"✗ Error: Invalid course name '{args.course_name}'")
            print(f"   Options: {list(course_map.keys())}")
            return

        full_course_name = course_map[args.course_name]
        result = process_file(llm, args.input_file, full_course_name, args.method_name)

        if result:
            print(f"\n{'='*60}")
            print("RESULTS:")
            print(f"{'='*60}")
            print(json.dumps(result, indent=2))
            # [JR] Honor --output_json here too (was batch-only); skip the
            # default so plain runs don't drop a stray file in CWD.
            if args.output_json != parser.get_default("output_json"):
                with open(args.output_json, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\n✓ Saved results → {args.output_json}")
        return

    # BATCH MODE (original behavior)
    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            results = json.load(f)
        print("Previous results loaded...")
        print(results)
    else:
        results = {
            "anlp": {"node_significance": {}, "triplet_accuracy": {}},
            "algo": {"node_significance": {}, "triplet_accuracy": {}},
            "sql": {"node_significance": {}, "triplet_accuracy": {}},
        }

    for method in sorted(os.listdir(args.input_root)):
        method_dir = os.path.join(args.input_root, method)
        if not os.path.isdir(method_dir):
            continue

        for path in glob.glob(os.path.join(method_dir, "*.jsonl")):
            fname = os.path.basename(path)
            model = fname.split("_")[-1].replace(".jsonl", "")
            course_code = fname.split("_")[1]

            print(f"\n{'='*60}")
            print(f"Processing: {fname}")
            print(f"Method: {method}")
            print(f"Model: {model}")
            print(f"Course: {course_code}")
            print(f"{'='*60}")

            if (
                (course_code in results)
                and (model in results[course_code]["node_significance"])
                and (method in results[course_code]["node_significance"][model])
            ):
                print("✓ Already evaluated - skipping")
                continue

            if course_code == "algo":
                course_name = "Efficient Algorithms and Intractable Problems"
            elif course_code == "anlp":
                course_name = "Advanced Topics in Natural Language Processing"
            elif course_code == "sql":
                course_name = "Database Systems"
            else:
                print(f"⚠ Unknown course code: {course_code} - skipping")
                continue

            # Load and parse JSONL
            data: List[dict] = []
            try:
                with open(path, "r") as f:
                    lines = f.readlines()
                    print(f"File has {len(lines)} lines")

                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"⚠ JSON error on line {i+1}: {e}")
                            continue

                    print(f"Successfully parsed {len(data)} valid JSON objects")

            except FileNotFoundError:
                print(f"✗ File not found: {path}")
                continue
            except Exception as e:
                print(f"✗ Error reading file: {e}")
                continue

            if not data:
                print("✗ No valid data loaded - file may be empty or malformed")
                print(f"   Check file: {path}")
                continue

            # Run evaluations
            try:
                print("\n→ Evaluating node significance...")
                ns = eval_node_significance(llm, data, course_name)
                print(f"  Result: {ns}")

                print("\n→ Evaluating triplet accuracy...")
                ta = eval_triplet_accuracy(llm, data, course_name)
                print(f"  Result: {ta}")

                # Store results
                results[course_code]["node_significance"].setdefault(model, {})[method] = ns
                results[course_code]["triplet_accuracy"].setdefault(model, {})[method] = ta

                # Save after each evaluation
                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\n✓ Saved results → {args.output_json}")

            except Exception as e:
                print(f"\n✗ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()