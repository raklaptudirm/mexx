//! lru implements a Least Recently Used cache for Nodes. This is used to store
//! the nodes of a Tree, allowing arbitrarily long searches as unused memory is
//! continuously freed by the cache.
use std::mem;

use derive_more::{Deref, DerefMut};
use derive_new::new;

use super::{Edge, Node};

/// Cache is a Least Recently Used (LRU) Cache for [Nodes](Node), which allows
/// the search tree to utilize limited memory efficiently.
#[derive(Clone)]
pub struct Cache {
    root_edge: Edge, // Root edge of the whole tree.
    map: Vec<Entry>, // Backing storage of the cache.

    cap: usize,

    void: i32, // Pointer to the first void in the cache.
    head: i32, // Pointer to the most recently used entry.
    tail: i32, // Pointer to the least recently used entry.
}

impl Cache {
    /// new_mib creates a new Cache with the given number of mebibytes of
    /// capacity for storing Nodes. Note that the actual memory usage will not
    /// be exactly the provided capacity as the memory used by edges cannot be
    /// determined but only guessed from the number of nodes.
    pub fn new_mib(mib: usize) -> Cache {
        // The / 60 is an adjustment to allow space for the edges.
        Cache::new(1024 * 1024 * mib / mem::size_of::<Entry>() / 60)
    }

    /// new creates a new Cache with the given capacity for storing Nodes.
    pub fn new(cap: usize) -> Cache {
        Cache {
            map: vec![Entry::new(); cap],
            root_edge: Edge::new(ataxx::Move::NULL),
            cap,
            void: 0,  // The first (0) entry is currently a void.
            head: -1, // Currently there is no most recently used entry.
            tail: -1, // Currently there is no least recently used entry.
        }
    }
}

impl Cache {
    /// promote makes the given Entry the most recently used one.
    pub fn promote(&mut self, ptr: i32) {
        self.detach(ptr);
        self.attach(ptr);
    }

    /// attach adds the given entry to the non-void list and makes it the head.
    pub fn attach(&mut self, ptr: i32) {
        // Update old head's links if there is one.
        if self.head != -1 {
            self.node_mut(self.head).prev = ptr;
        }

        // Create a copy of the pointer to the old head and make the attached
        // pointer the new head (most recently used entry) of the cache.
        let head_ptr = self.head;
        self.head = ptr;

        // Update the new head's links.
        let node = self.node_mut(ptr);
        node.next = head_ptr;
        node.prev = -1;
    }

    /// push adds the given Node to the cache as its head.
    pub fn push(&mut self, val: Node) -> i32 {
        // Find an Entry to store the node in.
        let node_ptr = if (self.void as usize) < self.cap {
            // Void Entry found, so we will use that. This pointer will no
            // longer be empty so, update it to the next void spot, which due
            // to the way the cache works will be next pointer.
            self.void += 1;
            // Return the pointer to the void Entry.
            self.void - 1
        } else {
            // No void spots left, so purge the least recently used entry (also
            // called the tail of the cache) and use that spot for storage.
            self.remove_lru()
        };

        // Update the value of the entry and attach it to the cache.
        self.node_mut(node_ptr).val = val;
        self.attach(node_ptr);

        // Return the pointer to the newly added entry.
        node_ptr
    }

    /// detach removes the given entry from the cache, but keeps its data. To
    /// also remove the entry's data, use [`Self::remove_lru`].
    fn detach(&mut self, ptr: i32) {
        let node = self.node(ptr);
        let (prev_ptr, next_ptr) = (node.prev, node.next);

        // Update the links for the Node's predecessor, if any.
        if prev_ptr != -1 {
            self.node_mut(prev_ptr).next = next_ptr;
        } else {
            // If there is no predecessor, this was the head node.
            // Update the pointer to the head node to the successor.
            self.head = next_ptr;
        }

        // Update the links for the Node's successor, if any.
        if next_ptr != -1 {
            self.node_mut(next_ptr).prev = prev_ptr;
        } else {
            // If there is no successor, this was the tail node.
            // Update the pointer to the tail node to the predecessor.
            self.tail = prev_ptr;
        }
    }

    /// remove_lru purges the data of the Least Recently Used Entry, removes all
    /// links to it, and detaches it from the used LRU cache space. It returns
    /// the pointer to the purged Entry.
    fn remove_lru(&mut self) -> i32 {
        let tail = self.tail;
        let node = self.node(tail);

        // Remove all links to the detached Entry.
        let (parent_node, parent_edge) = (node.parent_node, node.parent_edge);
        self.edge_mut(parent_node, parent_edge).ptr = -1;

        // Detach the Entry.
        self.detach(tail);

        // Return the pointer to the purged LRU entry.
        tail
    }
}

impl Cache {
    /// node returns a reference to the Entry at the given pointer.
    pub fn node(&self, ptr: i32) -> &Entry {
        &self.map[ptr as usize]
    }

    /// node_mut returns a mutable reference to the Entry at the given pointer.
    pub fn node_mut(&mut self, ptr: i32) -> &mut Entry {
        &mut self.map[ptr as usize]
    }

    pub fn edge(&self, parent: i32, edge_ptr: i32) -> &Edge {
        if parent == -1 {
            &self.root_edge
        } else {
            self.node(parent).edge(edge_ptr)
        }
    }

    pub fn edge_mut(&mut self, parent: i32, edge_ptr: i32) -> &mut Edge {
        if parent == -1 {
            &mut self.root_edge
        } else {
            self.node_mut(parent).edge_mut(edge_ptr)
        }
    }
}

/// Entry is one of the entries in the LRU [Cache]. Externally, it is mainly
/// used by dereferencing it into a [Node] instead of directly using it.
#[derive(Clone, Deref, DerefMut, new)]
pub struct Entry {
    #[deref]
    #[deref_mut]
    #[new(value = "Default::default()")]
    val: Node,
    #[new(value = "-1")]
    prev: i32,
    #[new(value = "-1")]
    next: i32,
}
