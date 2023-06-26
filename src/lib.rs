#![warn(non_camel_case_types, non_upper_case_globals, unused_qualifications)]
#![allow(clippy::unreadable_literal, clippy::bool_comparison)]

use std::{cmp, u64};
use std::convert::TryFrom;
use std::hash::Hasher;
use std::io::{Cursor, Read, Write};

use bit_vec::BitVec;

use base64::{Engine as _, engine::general_purpose};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use fasthash::{FastHasher, HasherExt, murmur3::Hasher128_x64, murmur3::Hasher128_x86};

pub mod reexports {
    #[cfg(feature = "random")]
    pub use ::getrandom;
    pub use bit_vec;
    pub use siphasher;
    #[cfg(feature = "serde")]
    pub use siphasher::reexports::serde;
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub enum HashStrategy {
    Murmur128Mitz32,

    #[default]
    Murmur128Mitz64,
}

impl HashStrategy {
    fn init_bloom_hash(&self, item: &[u8]) -> [i64; 2] {
        let mut hashes = [0i64, 0i64];

        match self {
            HashStrategy::Murmur128Mitz32 => {
                let mut h = Hasher128_x86::new();
                h.write(item);
                let hash = h.finish_ext();

                hashes[0] = (hash & 0x00000000ffffffff) as i64;
                hashes[1] = ((hash & 0xffffffff00000000) >> 32) as i64;
            }

            HashStrategy::Murmur128Mitz64 => {
                let mut h = Hasher128_x64::new();
                h.write(item);
                let hash = h.finish_ext();

                hashes[0] = (hash & 0x0000000000000000ffffffffffffffff) as i64;
                hashes[1] = ((hash & 0xffffffffffffffff0000000000000000) >> 64) as i64;
            }
        };

        // println!("hashes {}, {}", hashes[0], hashes[1]);

        return hashes;
    }

    fn bloom_hash(&self, hashes: &mut [i64; 2], k_i: u8) -> u64
    {
        if k_i > 0 {
            hashes[0] = hashes[0].wrapping_add(hashes[1]);
        }

        return (hashes[0] & i64::MAX) as u64;
    }


    pub fn dumps(&self, result: &mut Vec<u8>) {
        match self {
            HashStrategy::Murmur128Mitz32 => {
                result.write_u8(0).unwrap();
            }
            HashStrategy::Murmur128Mitz64 => {
                result.write_u8(1).unwrap();
            }
        }
    }

    fn from_bytes(cursor: &mut Cursor<&[u8]>) -> Result<HashStrategy, &'static str> {
        let strategy_ordinal = cursor.read_u8().unwrap();
        match strategy_ordinal {
            0 => Ok(HashStrategy::Murmur128Mitz32),
            1 => Ok(HashStrategy::Murmur128Mitz64),
            _ => Err("Invalid strategy ordinal"),
        }
    }
}

/// Bloom filter structure
#[derive(Clone, Debug)]
pub struct Bloom {
    bit_vec: BitVec<u64>,
    bitmap_bits: u64,
    hash_num: u8,
    hash_strategy: HashStrategy,
}

impl Bloom {
    /// Create a new bloom filter structure.
    /// bitmap_size is the size in bytes (not bits) that will be allocated in
    /// memory items_count is an estimation of the maximum number of items
    /// to store. seed is a random value used to generate the hash
    /// functions.
    pub fn new_with_hash_strategy(bitmap_size: usize, items_count: usize, hash_strategy: HashStrategy) -> Self {
        assert!(bitmap_size > 0 && items_count > 0);
        let bitmap_bits = u64::try_from(bitmap_size)
            .unwrap()
            .checked_mul(8u64)
            .unwrap();
        let hash_num = Self::optimal_k_num(bitmap_bits, items_count);

        // shailendra: align bit vec size to 8 bytes chunks
        let bit_vec_size = (bitmap_bits as f64 / 64.0).ceil() as usize * 64;
        let bit_vec = BitVec::from_elem(bit_vec_size, false);

        Self {
            bit_vec,
            bitmap_bits,
            hash_num,
            hash_strategy,
        }
    }

    /// Create a new bloom filter structure.
    /// bitmap_size is the size in bytes (not bits) that will be allocated in
    /// memory items_count is an estimation of the maximum number of items
    /// to store.
    #[cfg(feature = "random")]
    pub fn new(bitmap_size: usize, items_count: usize) -> Self {
        Self::new_with_hash_strategy(bitmap_size, items_count, HashStrategy::default())
    }

    /// Create a new bloom filter structure.
    /// items_count is an estimation of the maximum number of items to store.
    /// fp_p is the wanted rate of false positives, in ]0.0, 1.0[
    #[cfg(feature = "random")]
    pub fn new_for_fp_rate(items_count: usize, fp_p: f64) -> Self {
        let bitmap_size = Self::compute_bitmap_size(items_count, fp_p);
        Bloom::new_with_hash_strategy(bitmap_size, items_count, HashStrategy::default())
    }

    /// Create a new bloom filter structure.
    /// items_count is an estimation of the maximum number of items to store.
    /// fp_p is the wanted rate of false positives, in ]0.0, 1.0[
    pub fn new_for_fp_rate_with_hash_strategy(items_count: usize, fp_p: f64, hash_strategy: HashStrategy) -> Self {
        let bitmap_size = Self::compute_bitmap_size(items_count, fp_p);
        Bloom::new_with_hash_strategy(bitmap_size, items_count, hash_strategy)
    }

    /// Create a bloom filter structure from a previous state given as a
    /// `ByteVec` structure. The state is assumed to be retrieved from an
    /// existing bloom filter.
    pub fn from_bit_vec(
        bit_vec: BitVec<u64>,
        bitmap_bits: u64,
        hash_num: u8,
        hash_strategy: HashStrategy,
    ) -> Self {
        Self {
            bit_vec,
            bitmap_bits,
            hash_num,
            hash_strategy,
            // _phantom: PhantomData,
        }
    }

    /// Create a bloom filter structure with an existing state given as a byte
    /// array. The state is assumed to be retrieved from an existing bloom
    /// filter.
    pub fn from_existing(
        bytes: &[u8],
        bitmap_bits: u64,
        hash_num: u8,
        hash_strategy: HashStrategy,
    ) -> Self {
        Self::from_bit_vec(BitVec::from_bytes(bytes), bitmap_bits, hash_num, hash_strategy)
    }

    pub fn compute_num_bits(items_count: usize, mut fp_p: f64) -> usize {
        assert!(items_count > 0);
        if fp_p == 0.0 {
            fp_p = 2_f64.powf(-1074.0)
        }

        assert!(fp_p > 0.0 && fp_p < 1.0);
        let log2 = std::f64::consts::LN_2;
        let log2_2 = log2 * log2;

        // TODO (Shailendra): python code takes floor here, we can make the same mistake... but ceil is better, in my opinion.
        ((items_count as f64) * f64::ln(fp_p) / (-1.0 * log2_2)).ceil() as usize
    }

    /// Compute a recommended bitmap size for items_count items
    /// and a fp_p rate of false positives.
    /// fp_p obviously has to be within the ]0.0, 1.0[ range.
    pub fn compute_bitmap_size(items_count: usize, fp_p: f64) -> usize {
        let bit_count = Self::compute_num_bits(items_count, fp_p);
        (bit_count as f64 / 8.0).ceil() as usize
    }

    /// Record the presence of an item.
    pub fn set(&mut self, item: &[u8])
    {
        // println!("bit map bits: {}", self.bitmap_bits);

        let mut hashes = self.hash_strategy.init_bloom_hash(item);
        for k_i in 0..self.hash_num {
            let bit_offset = (self.hash_strategy.bloom_hash(&mut hashes, k_i) % self.bitmap_bits) as usize;
            // println!("for k_i={} bit_offset={}", k_i, bit_offset);

            self.bit_vec.set(bit_offset, true);
        }
    }

    /// Check if an item is present in the set.
    /// There can be false positives, but no false negatives.
    pub fn check(&self, item: &[u8]) -> bool
    {
        let mut hashes = self.hash_strategy.init_bloom_hash(item);
        for k_i in 0..self.hash_num {
            let bit_offset = (self.hash_strategy.bloom_hash(&mut hashes, k_i) % self.bitmap_bits) as usize;
            // println!("for k_i={} bit_offset={}", k_i, bit_offset);

            if self.bit_vec.get(bit_offset).unwrap() == false {
                return false;
            }
        }
        true
    }

    /// Record the presence of an item in the set,
    /// and return the previous state of this item.
    pub fn check_and_set(&mut self, item: &[u8]) -> bool
    {
        let mut hashes = self.hash_strategy.init_bloom_hash(item);
        let mut found = true;
        for k_i in 0..self.hash_num {
            let bit_offset = (self.hash_strategy.bloom_hash(&mut hashes, k_i) % self.bitmap_bits) as usize;
            if self.bit_vec.get(bit_offset).unwrap() == false {
                found = false;
                self.bit_vec.set(bit_offset, true);
            }
        }
        found
    }

    /// Return the bitmap as a vector of bytes
    pub fn bitmap(&self) -> Vec<u8> {
        self.bit_vec.to_bytes()
    }

    /// Return the bitmap as a "BitVec" structure
    pub fn bit_vec(&self) -> &BitVec<u64> {
        &self.bit_vec
    }

    /// Return the number of bits in the filter
    pub fn number_of_bits(&self) -> u64 {
        self.bitmap_bits
    }

    /// Return the number of hash functions used for `check` and `set`
    pub fn number_of_hash_functions(&self) -> u8 {
        self.hash_num
    }

    pub fn hash_strategy(&self) -> &HashStrategy {
        &self.hash_strategy
    }

    #[allow(dead_code)]
    pub fn optimal_k_num(bitmap_bits: u64, items_count: usize) -> u8 {
        let m = bitmap_bits as f64;
        let n = items_count as f64;
        let k_num = (m / n * f64::ln(2.0f64)).ceil() as u8;
        cmp::max(k_num, 1)
    }

    /// Clear all of the bits in the filter, removing all keys from the set
    pub fn clear(&mut self) {
        self.bit_vec.clear()
    }

    /// Set all of the bits in the filter, making it appear like every key is in the set
    pub fn fill(&mut self) {
        self.bit_vec.set_all()
    }

    /// Test if there are no elements in the set
    pub fn is_empty(&self) -> bool {
        !self.bit_vec.any()
    }

    pub fn dumps(&self) -> Vec<u8> {
        // Serial form:
        // 1 signed byte for the strategy
        // 1 unsigned byte for the number of hash functions
        // 1 big endian int, the number of longs in our bitset
        // N big endian longs of our bitset
        let mut result = Vec::new();
        self.hash_strategy.dumps(&mut result);
        result.write_u8(self.hash_num).unwrap();

        println!("bit vec len: {}", self.bit_vec.len());
        println!("bit vec bits: {}", self.bitmap_bits);
        println!("actual length of bit vec: {}", self.bit_vec.to_bytes().len());

        let num_u64 = (self.bit_vec.len() as f64 / 64.0).ceil() as u32;
        println!("num_u64 in dumps: {}", num_u64);
        result.write_u32::<BigEndian>(num_u64).unwrap();

        // let data = &self.bit_vec.to_bytes();
        // for chunk in data.chunks(8) {
        //     println!("chunk in dumps: {:?}", chunk);
        //     let rev_chunk: Vec<_> = chunk.iter().rev().cloned().collect();
        //
        //     println!("reverse chunk in dumps: {:?}", rev_chunk);
        //     result.write(&rev_chunk).unwrap();
        // }

        for i in 0_usize..num_u64 as usize {
            let chunk = &self.bit_vec[i..i + 64];
            println!("chunk in dumps: {:?}", chunk);
            let rev_chunk = chunk.reverse();

            println!("reverse chunk in dumps: {:?}", rev_chunk);
            result.write(&rev_chunk).unwrap();
        }

        // result.write(&self.bit_vec.to_bytes()).unwrap();

        // self.bit_vec.to_bytes()
        //     .chunks_exact(8)
        //     .map(|bytes| u64::from_ne_bytes(bytes.try_into().unwrap()));
        //
        // for chunk in self.bit_vec.to_bytes().chunks_exact(8) {
        //     result.write_u64::<BigEndian>()
        //     let reversed: Vec<u8> = chunk.iter().rev().copied().collect();
        // }

        result
    }

    pub fn dumps_to_hex(&self) -> String {
        self.dumps().iter().map(|byte| format!("{:02x}", byte)).collect()
    }

    pub fn dumps_to_base64(&self) -> String {
        general_purpose::STANDARD_NO_PAD.encode(&self.dumps())
    }

    pub fn from_bytes(array: &[u8]) -> Result<Bloom, &'static str> {
        let mut cursor = Cursor::new(array);
        let strategy = HashStrategy::from_bytes(&mut cursor)?;
        let hash_num = cursor.read_u8().unwrap();
        let num_u64 = cursor.read_u32::<BigEndian>().unwrap() as u64;

        println!("Num Longs in read: {}", num_u64);

        let mut data = Vec::new(); // vec![0u8; num_u64 as usize * 8];
        for _ in 0..num_u64 {
            let mut chunk = vec![0u8; 8];
            cursor.read_exact(&mut chunk).unwrap();
            chunk.reverse();
            println!("chunk: {:?}", chunk);

            data.write(&chunk).unwrap();
            // let val = cursor.read_u64::<BigEndian>().unwrap();
        }

        println!("data: {:?}", data);

        // cursor.read_exact(&mut data).unwrap();

        // Create and setup the instance
        let instance = Bloom::from_existing(&data, num_u64, hash_num, strategy);

        Ok(instance)
    }

    pub fn from_hex(hex_str: &str) -> Result<Bloom, &'static str> {
        let bytes = hex::decode(hex_str).unwrap();
        Bloom::from_bytes(&bytes)
    }

    pub fn from_base64(base64_encoded: &str) -> Result<Bloom, &'static str> {
        let bytes = general_purpose::STANDARD_NO_PAD.decode(base64_encoded).unwrap();
        Bloom::from_bytes(&bytes)
    }
}