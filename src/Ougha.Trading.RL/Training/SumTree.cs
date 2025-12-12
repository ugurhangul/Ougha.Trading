namespace Ougha.Trading.RL.Training;

/// <summary>
/// Sum tree data structure for O(log n) priority sampling.
/// </summary>
internal class SumTree(int capacity)
{
    private readonly float[] _tree = new float[2 * capacity - 1];

    public float Total => _tree[0];
    
    public float Min
    {
        get
        {
            var min = float.MaxValue;
            for (var i = capacity - 1; i < _tree.Length; i++)
            {
                if (_tree[i] > 0 && _tree[i] < min)
                    min = _tree[i];
            }
            return min > 0 ? min : 1e-6f;
        }
    }

    public void Update(int dataIndex, float priority)
    {
        var treeIndex = dataIndex + capacity - 1;
        var change = priority - _tree[treeIndex];
        _tree[treeIndex] = priority;

        while (treeIndex > 0)
        {
            treeIndex = (treeIndex - 1) / 2;
            _tree[treeIndex] += change;
        }
    }

    public int Find(float value)
    {
        var idx = 0;
        
        while (idx < capacity - 1)
        {
            var left = 2 * idx + 1;
            var right = left + 1;
            
            if (value <= _tree[left])
            {
                idx = left;
            }
            else
            {
                value -= _tree[left];
                idx = right;
            }
        }
        
        return idx - (capacity - 1);
    }

    public void Clear()
    {
        Array.Clear(_tree, 0, _tree.Length);
    }
}
