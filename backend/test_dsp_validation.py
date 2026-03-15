import os
from app.validator import DSPValidator

# Initialize validator
validator = DSPValidator()

# Your 5 test songs - UPDATE PATHS HERE
test_songs = [
    # (path, bpm, key, genre)
    ('uploads/test_songs/daft_punk_one_more_time.mp3', 123, 'A minor', 'House'),
    ('uploads/test_songs/drake_gods_plan.mp3', 77, 'Bb major', 'Hip-Hop'),
    ('uploads/test_songs/ed_sheeran_shape_of_you.mp3', 96, 'C# minor', 'Pop'),
    ('uploads/test_songs/the_killers_mr_brightside.mp3', 148, 'D major', 'Rock'),
    ('uploads/test_songs/deadmau5_strobe.mp3', 128, 'Db major', 'Electronic'),
]

print("🎵 Starting DSP Validation Test")
print("="*70)

for path, bpm, key, genre in test_songs:
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(path)}")
    print(f"Expected: {bpm} BPM, {key}")
    print(f"{'='*70}")
    
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        print(f"   Looking for files in: {os.path.abspath('app/uploads/test_songs/')}")
        continue
    
    try:
        # Test BPM
        validator.test_bpm(path, bpm, genre)
        
        # Test Key (skip if essentia not installed)
        try:
            validator.test_key(path, key)
        except ImportError:
            print("   ⚠️ Skipping key test (essentia not installed)")
        except Exception as e:
            print(f"   ⚠️ Key test error: {e}")
        
        # Test Loudness
        validator.test_loudness(path)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Export results
print(f"\n{'='*70}")
print("📊 Exporting Results...")
print(f"{'='*70}")
validator.export_results()

print("\n✅ Validation complete!")
print("📁 Check validation_results/ folder for CSV files")
