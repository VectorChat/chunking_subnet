import express, { Request, Response } from 'express';
import { exec } from 'child_process';
import fs from 'fs/promises';

const app = express();
app.use(express.json());

const CONFIG_FILE = '/data/ipfs-cluster/service.json';

app.put('/update-trusted-peers', async (req: Request, res: Response) => {
  try {
    const { trustedPeers } = req.body;

    if (!Array.isArray(trustedPeers)) {
      return res.status(400).json({ error: 'trustedPeers must be an array' });
    }

    const config = JSON.parse(await fs.readFile(CONFIG_FILE, 'utf-8'));

    if (JSON.stringify(config.consensus.crdt.trusted_peers) !== JSON.stringify(trustedPeers)) {
      config.consensus.crdt.trusted_peers = trustedPeers;
      await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2));

      // Restart IPFS Cluster
      exec('pm2 restart ipfs-cluster-service', (error) => {
        if (error) {
          console.error('Error restarting IPFS Cluster:', error);
          return res.status(500).json({ error: 'Failed to restart IPFS Cluster' });
        }
        res.status(200).json({ message: 'Trusted peers updated and IPFS Cluster restarted' });
      });
    } else {
      res.status(200).json({ message: 'No changes in trusted peers' });
    }

  } catch (error) {
    console.error('Error updating trusted peers:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/status', (req, res) => {
  exec('pm2 jlist', (error, stdout) => {
    if (error) {
      console.error('Error getting PM2 status:', error);
      return res.status(500).json({ error: 'Failed to get service status' });
    }

    try {
      const processes = JSON.parse(stdout);
      const ipfsStatus = processes.find((p: any) => p.name === 'ipfs')?.pm2_env?.status;
      const ipfsClusterStatus = processes.find((p: any) => p.name === 'ipfs-cluster-service')?.pm2_env?.status;

      res.status(200).json({
        ipfs: ipfsStatus === 'online',
        ipfsCluster: ipfsClusterStatus === 'online',
      });
    } catch (parseError) {
      console.error('Error parsing PM2 output:', parseError);
      res.status(500).json({ error: 'Failed to parse service status' });
    }
  });
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Custom API server running on port ${PORT}`);
});