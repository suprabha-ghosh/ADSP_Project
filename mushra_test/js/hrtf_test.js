// Configuration setup
function setupMushraTest() {
    var config = {
        "TestName": "HRTF Comparison Test",
        "RateScalePng": "img/scale_abs.png",
        "RateScaleBgPng": "img/scale_abs_background.png",
        "RateMinValue": 0,
        "RateMaxValue": 100,
        "RateDefaultValue": 0,
        "ShowFileIDs": false,
        "ShowResults": false,
        "AudioRoot": "",  // Base path for audio files
        "Testsets": []
    };

    // Create test sets for each participant
    for (let i = 1; i <= 6; i++) {
        config.Testsets.push({
            "Name": `Participant ${i}`,
            "TestID": `test_${i}`,
            "Files": {
                "Reference": "",  // No reference in this case
                "model_hrtf": `model_hrtf${i}.wav`,
                "benchmark_hrtf": `benchmark_hrtf_person${i}.wav`
            }
        });
    }

    return config;
}

// Audio loading and management
let currentParticipant = 0;
let audioContext = null;
let audioBuffers = {};

// Initialize audio context
function initAudioContext() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        return true;
    } catch(e) {
        console.error("Web Audio API is not supported in this browser");
        return false;
    }
}

// Load audio file
async function loadAudio(url) {
    try {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        return audioBuffer;
    } catch(e) {
        console.error(`Error loading audio file: ${url}`, e);
        return null;
    }
}

// Load audio files for current participant
async function loadParticipantAudio(participantIndex) {
    if (!audioContext) {
        if (!initAudioContext()) return;
    }

    const testset = pageConfig.Testsets[participantIndex];
    if (!testset) return;

    // Clear previous buffers
    audioBuffers = {};

    try {
        // Load model HRTF
        audioBuffers.model_hrtf = await loadAudio(testset.Files.model_hrtf);
        
        // Load benchmark HRTF
        audioBuffers.benchmark_hrtf = await loadAudio(testset.Files.benchmark_hrtf);

        console.log(`Loaded audio files for participant ${participantIndex + 1}`);
    } catch(e) {
        console.error("Error loading audio files:", e);
    }
}

// Play audio function
let currentSource = null;

function playAudio(audioId) {
    if (!audioContext || !audioBuffers[audioId]) {
        console.error(`Audio not loaded for ${audioId}`);
        return;
    }

    // Stop any currently playing audio
    if (currentSource) {
        currentSource.stop();
        currentSource = null;
    }

    // Create new audio source
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffers[audioId];
    source.connect(audioContext.destination);
    
    // Store current source
    currentSource = source;

    // Play the audio
    source.start(0);

    // Update play button state
    updatePlayButton(audioId, true);

    // Handle playback end
    source.onended = () => {
        currentSource = null;
        updatePlayButton(audioId, false);
    };
}

// Update play button appearance
function updatePlayButton(audioId, isPlaying) {
    const button = document.querySelector(`button[data-audio-id="${audioId}"]`);
    if (button) {
        const icon = button.querySelector('i');
        icon.className = isPlaying ? 'fas fa-pause' : 'fas fa-play';
    }
}

// Stop audio playback
function stopAudio() {
    if (currentSource) {
        currentSource.stop();
        currentSource = null;
    }
}

// Handle next participant
function nextParticipant() {
    if (!validateName()) return;

    // Save current ratings
    const ratings = {
        name: document.getElementById('participantName').value,
        modelRating: document.querySelector('input[data-audio-id="model_hrtf"]').value,    // Changed this line
        benchmarkRating: document.querySelector('input[data-audio-id="benchmark_hrtf"]').value    // Changed this line
    };
    
    // Log the ratings to verify values are captured
    console.log("Saving ratings:", ratings);
    
    participants.push(ratings);

    // Stop any playing audio
    document.querySelectorAll('audio').forEach(audio => {
        audio.pause();
        audio.currentTime = 0;
    });

    // Reset play buttons and progress bars
    document.querySelectorAll('.play-pause-button i').forEach(icon => {
        icon.className = 'fas fa-play';
    });
    document.querySelectorAll('.progress-bar').forEach(bar => {
        bar.style.width = '0%';
    });

    currentParticipant++;
    if (currentParticipant < 6) {
        // Load next participant's audio files
        document.getElementById('model_hrtf').src = `audio/model_hrtf${currentParticipant + 1}.wav`;
        document.getElementById('benchmark_hrtf').src = `audio/benchmark_hrtf_person${currentParticipant + 1}.wav`;
        
        // Reset form
        document.getElementById('participantName').value = '';
        document.querySelectorAll('.rating-slider').forEach(slider => {
            slider.value = 0;
            updateRatingDisplay(slider, `${slider.dataset.audioId}_value`);
        });
    } else {
        showResults();
    }
}

function showResults() {
    const resultsDiv = document.getElementById('results');
    const resultsBody = document.getElementById('resultsBody');
    resultsDiv.classList.remove('hidden');

    // Clear previous results
    resultsBody.innerHTML = '';

    // Add all participants' results
    participants.forEach((p, index) => {
        const row = resultsBody.insertRow();
        row.insertCell(0).textContent = p.name;
        row.insertCell(1).textContent = p.modelRating;
        row.insertCell(2).textContent = p.benchmarkRating;
    });
}

function downloadResults() {
    // First, verify we have data
    console.log("Participants data:", participants);

    let csv = 'Participant,Model HRTF Rating,Benchmark HRTF Rating\n';
    participants.forEach(p => {
        // Add validation to ensure we have values
        const modelRating = p.modelRating || '0';
        const benchmarkRating = p.benchmarkRating || '0';
        csv += `${p.name},${modelRating},${benchmarkRating}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', 'mushra_test_results.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Initialize the test
window.onload = function() {
    pageConfig = setupMushraTest();
    loadParticipantAudio(currentParticipant);
};
