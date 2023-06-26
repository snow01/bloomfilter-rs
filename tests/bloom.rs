use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use base64::{Engine as _, engine::general_purpose};
use getrandom::getrandom;
use rand::Rng;

use bloomfilter::Bloom;

#[allow(dead_code)]
fn guava_file_dir() -> PathBuf {
    PathBuf::from(file!()) // This is equivalent to __file__ in Python
        .parent()
        .unwrap() // This is equivalent to os.path.dirname
        .join("guava_dump_files") // This is equivalent to os.path.join
}

#[allow(dead_code)]
fn read_data(filename: &str) -> std::io::Result<Vec<u8>> {
    let mut path = guava_file_dir();
    path.push(filename);
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

#[test]
#[cfg(feature = "random")]
fn bloom_test_set() {
    let mut bloom = Bloom::new(10, 80);
    let mut k = vec![0u8, 16];
    getrandom(&mut k).unwrap();
    assert_eq!(bloom.check(&k), false);
    bloom.set(&k);
    assert_eq!(bloom.check(&k), true);
}

#[test]
#[cfg(feature = "random")]
fn bloom_test_check_and_set() {
    let mut bloom = Bloom::new(10, 80);
    let mut k = vec![0u8, 16];
    getrandom(&mut k).unwrap();
    assert_eq!(bloom.check_and_set(&k), false);
    assert_eq!(bloom.check_and_set(&k), true);
}

#[test]
#[cfg(feature = "random")]
fn bloom_test_clear() {
    let mut bloom = Bloom::new(10, 80);
    let mut k = vec![0u8, 16];
    getrandom(&mut k).unwrap();
    bloom.set(&k);
    assert_eq!(bloom.check(&k), true);
    bloom.clear();
    assert_eq!(bloom.check(&k), false);
}

#[test]
#[cfg(feature = "random")]
fn bloom_test_load() {
    let mut original = Bloom::new(10, 80);
    let mut k = vec![0u8, 16];
    getrandom(&mut k).unwrap();
    original.set(&k);
    assert_eq!(original.check(&k), true);

    let cloned = Bloom::from_existing(
        &original.bitmap(),
        original.number_of_bits(),
        original.number_of_hash_functions(),
        original.hash_strategy().clone(),
    );
    assert_eq!(cloned.check(&k), true);
}

//
// Ported test functions from python implementation
//
#[test]
#[cfg(feature = "random")]
fn test_num_of_bits() {
    let test_cases = [(500, 0.01, 4793, 4800, 7), (500, 0.000001, 14378, 14400, 20), (10, 0.01, 96, 128, 7)];
    for (i, case) in test_cases.iter().enumerate() {
        let actual_bits = Bloom::compute_num_bits(case.0, case.1);
        assert_eq!(actual_bits, case.2, "For case {} got num bits={} but expected={}", i, actual_bits, case.2);

        let bloom_filter = Bloom::new_for_fp_rate(case.0, case.1);
        let bit_vec_size = bloom_filter.number_of_bits();
        assert_eq!(bit_vec_size, case.3, "For case {} got bit vec size={} but expected={}", i, bit_vec_size, case.3);

        let num_hash_functions = bloom_filter.number_of_hash_functions();
        assert_eq!(num_hash_functions, case.4, "For case {} got num hash functions={} but expected={}", i, num_hash_functions, case.3);
    }
}

#[test]
#[cfg(feature = "random")]
fn test_basic_functionality() {
    let mut bloom_filter = Bloom::new_for_fp_rate(10000000, 0.001);

    for i in 0_i32..200 {
        let item = i.to_be_bytes();
        bloom_filter.set(&item);
    }

    for i in 0_i32..200 {
        let item = i.to_be_bytes();
        assert!(
            bloom_filter.check(&item),
            "Number {} is expected to be in bloomfilter", i
        );
    }

    for i in 200_i32..500 {
        let item = i.to_be_bytes();
        assert!(
            !bloom_filter.check(&item),
            "Number {} is NOT expected to be in bloomfilter", i
        );
    }

    let mut bloom_filter = Bloom::new_for_fp_rate(10000000, 0.001);

    let words = vec!["hello", "world", "bloom", "filter"];
    for word in &words {
        let item = word.as_bytes();
        bloom_filter.set(item);
    }

    for word in &words {
        let item = word.as_bytes();

        assert!(
            bloom_filter.check(item),
            "Word '{}' is expected to be in bloomfilter", word
        );
    }

    let not_exist_word = "not_exist".as_bytes();
    assert!(
        !bloom_filter.check(not_exist_word),
        "Word 'not_exist' is expected to be in bloomfilter"
    );
}


#[test]
#[cfg(feature = "random")]
fn test_dumps() {
    let mut bloom_filter = Bloom::new_for_fp_rate(300, 0.0001);
    for i in 0_i32..100 {
        let item = i.to_be_bytes();
        bloom_filter.set(&item);
    }
    let byte_array = bloom_filter.dumps();
    let new_filter = Bloom::from_bytes(&byte_array).unwrap();

    assert_eq!(
        new_filter.number_of_hash_functions(),
        bloom_filter.number_of_hash_functions(),
        "New filter's num of hash functions = {} is expected to be the same as old filter's = {}",
        new_filter.number_of_hash_functions(),
        bloom_filter.number_of_hash_functions(),
    );

    assert_eq!(
        new_filter.hash_strategy(),
        bloom_filter.hash_strategy(),
        "New filter's strategy = {:?} is expected to be the same as old filter's = {:?}",
        new_filter.hash_strategy(),
        bloom_filter.hash_strategy(),
    );

    assert_eq!(
        new_filter.bitmap(),
        bloom_filter.bitmap(),
        "New filter's data is expected to be the same as old filter's",
    );

    assert_eq!(
        new_filter.dumps(),
        byte_array,
        "New filter's dump = {} is expected to be the same as old filter's = {}",
        new_filter.dumps_to_hex(),
        general_purpose::STANDARD_NO_PAD.encode(&byte_array)
    );
}

#[test]
#[cfg(feature = "random")]
fn test_guava_compatibility_1() {
    let data = read_data("500_0_01_0_to_99_test.out").unwrap();
    let loaded_bloom_filter = Bloom::from_bytes(&data).unwrap();

    let mut new_bloom_filter = Bloom::new_for_fp_rate(500, 0.01);
    for i in 0_i32..100 {
        let item = i.to_le_bytes();

        new_bloom_filter.set(&item);
    }

    assert_eq!(
        new_bloom_filter.number_of_hash_functions(),
        loaded_bloom_filter.number_of_hash_functions(),
        "New filter's num of hash functions = {} is expected to be the same as loaded filter's = {}",
        new_bloom_filter.number_of_hash_functions(),
        loaded_bloom_filter.number_of_hash_functions(),
    );

    assert_eq!(
        new_bloom_filter.hash_strategy(),
        loaded_bloom_filter.hash_strategy(),
        "New filter's strategy = {:?} is expected to be the same as loaded filter's = {:?}",
        new_bloom_filter.hash_strategy(),
        loaded_bloom_filter.hash_strategy(),
    );

    assert_eq!(
        new_bloom_filter.bitmap(),
        loaded_bloom_filter.bitmap(),
        "New filter's data is expected to be the same as loaded filter's",
    );

    assert_eq!(
        new_bloom_filter.dumps_to_hex(),
        loaded_bloom_filter.dumps_to_hex(),
        "New filter's dump = {} is expected to be the same as loaded filter's = {}",
        new_bloom_filter.dumps_to_hex(),
        loaded_bloom_filter.dumps_to_hex(),
    );

    for i in 0_i32..100 {
        let item = i.to_le_bytes();

        assert!(loaded_bloom_filter.check(&item), "Number {} is expected to be in bloomfilter", i);
    }
}

#[test]
#[cfg(feature = "random")]
fn test_guava_compatibility_2() {
    let data = read_data("100_0_001_0_to_49_test.out").unwrap();
    let loaded_bloom_filter = Bloom::from_bytes(&data).unwrap();

    let mut new_bloom_filter = Bloom::new_for_fp_rate(100, 0.001);
    for i in 0_i32..50 {
        let item = i.to_le_bytes();

        new_bloom_filter.set(&item);
    }

    assert_eq!(
        new_bloom_filter.number_of_hash_functions(),
        loaded_bloom_filter.number_of_hash_functions(),
        "New filter's num of hash functions = {} is expected to be the same as loaded filter's = {}",
        new_bloom_filter.number_of_hash_functions(),
        loaded_bloom_filter.number_of_hash_functions(),
    );

    assert_eq!(
        new_bloom_filter.hash_strategy(),
        loaded_bloom_filter.hash_strategy(),
        "New filter's strategy = {:?} is expected to be the same as loaded filter's = {:?}",
        new_bloom_filter.hash_strategy(),
        loaded_bloom_filter.hash_strategy(),
    );

    assert_eq!(
        new_bloom_filter.number_of_bits(),
        loaded_bloom_filter.number_of_bits(),
        "New filter's number of bits = {:?} is expected to be the same as loaded filter's = {:?}",
        new_bloom_filter.number_of_bits(),
        loaded_bloom_filter.number_of_bits(),
    );

    assert_eq!(
        new_bloom_filter.bitmap(),
        loaded_bloom_filter.bitmap(),
        "New filter's data is expected to be the same as loaded filter's",
    );

    assert_eq!(
        new_bloom_filter.dumps_to_hex(),
        loaded_bloom_filter.dumps_to_hex(),
        "New filter's dump = {} is expected to be the same as loaded filter's = {}",
        new_bloom_filter.dumps_to_hex(),
        loaded_bloom_filter.dumps_to_hex(),
    );

    for i in 0_i32..50 {
        let item = i.to_le_bytes();

        assert!(loaded_bloom_filter.check(&item), "Number {} is expected to be in bloomfilter", i);
    }
}

#[test]
#[cfg(feature = "random")]
fn test_dumps_to_hex() {
    let mut rng = rand::thread_rng();
    let mut bloom_filter = Bloom::new_for_fp_rate(500, 0.0001);

    for _ in 0..100 {
        let i: i64 = rng.gen_range(100000000..10000000000);
        let item = i.to_be_bytes();
        bloom_filter.set(&item);
    }

    let hex_string = bloom_filter.dumps_to_hex();

    let new_filter = Bloom::from_hex(&hex_string).unwrap();

    assert_eq!(new_filter.number_of_hash_functions(), bloom_filter.number_of_hash_functions(),
               "New filter's num of hash functions is expected to be the same as old filter's");

    // TODO: Need to compare other parameters of new and old filters

    let new_filter_dump = new_filter.dumps_to_hex();
    assert_eq!(new_filter_dump, hex_string,
               "New filter's dump is expected to be the same as old filter's");
}

#[test]
#[cfg(feature = "random")]
fn test_dumps_to_base64() {
    let mut rng = rand::thread_rng();
    let mut bloom_filter = Bloom::new_for_fp_rate(500, 0.0001);

    for _ in 0..100 {
        let i: i64 = rng.gen_range(100000000..10000000000);
        let item = i.to_be_bytes();

        bloom_filter.set(&item);
    }

    let base64_encoded = bloom_filter.dumps_to_base64();

    let new_filter = Bloom::from_base64(&base64_encoded).unwrap();

    assert_eq!(new_filter.number_of_hash_functions(), bloom_filter.number_of_hash_functions(),
               "New filter's num of hash functions is expected to be the same as old filter's");

    // TODO: Need to compare other parameters of new and old filters

    let new_filter_dump = new_filter.dumps_to_base64();
    assert_eq!(new_filter_dump, base64_encoded,
               "New filter's dump is expected to be the same as old filter's");
}
